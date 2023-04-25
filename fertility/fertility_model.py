import itertools
from typing import Optional, Dict, Tuple

import torch
from allennlp.common import Registrable
from allennlp.modules import FeedForward
from allennlp.nn.util import get_mask_from_sequence_lengths, get_lengths_from_binary_sequence_mask
from torch.nn import Module

import torch.nn.functional as F
from fertility.conv_utils import cumulative_sum, sum_of_rand_vars
from fertility.fertility_numba import fertility_layer_joint

def sinusoidal_pos_embedding(d_model: int, max_len: int = 5000, pos_offset: int = 0,
                             device: Optional[torch.device] = None):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1) + pos_offset
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class FertilityModel(Module, Registrable):
    def __init__(self, maximum_fertility: int, input_dim: int, positional_dim: int = 0, positional_mode: str = "sum",
                 sinusoidal: bool = False, max_length: Optional[int] = None):
        super(FertilityModel, self).__init__()
        self.maximum_fertility = maximum_fertility
        self.input_dim = input_dim
        self.positional_dim = positional_dim
        if sinusoidal:
            self.positional_embedding = torch.nn.Parameter(sinusoidal_pos_embedding(input_dim, self.maximum_fertility+1),
                                                           requires_grad=False)
        else:
            self.positional_embedding = torch.nn.Parameter(torch.randn([1, self.maximum_fertility+1, self.positional_dim]),
                                                           requires_grad=True)
        assert positional_mode is None or positional_mode in {"cat", "sum"}
        self.positional_mode = positional_mode

        self.max_length = max_length

        # torch.nn.init.normal_(self.positional_embedding)
        # torch.nn.init.uniform_(self.positional_embedding, -1, 1)


    def get_output_dim(self) -> int:
        if self.positional_mode == "sum":
            return self.input_dim
        return self.input_dim + self.positional_dim


    def get_input_dim(self) -> int:
        return self.input_dim

    def compute_marginal_alignment(self, inputs: torch.Tensor, input_mask: torch.Tensor, target_mask: torch.Tensor) \
            -> Dict[str, torch.Tensor]:
        """
        Computes a marginal monotonic alignment based on a fertility step.
        Also computes marginal positions.
        :param input_mask: bool tensor of shape (batch_size, input_seq_len), where 0 indicates padding.
        :param inputs: shape (batch_size, input_seq_len, dim)
        :param target_mask: (batch_size, output_seq_len)
        :return: a dictionary with those values.
        """
        raise NotImplementedError()



    def top_k_lengths(self, inputs_q: torch.Tensor, input_mask: torch.Tensor, k: int, length_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the top k total positive lengths
        :param inputs_q: shape (batch_size, input_seq_len, encoding dim)
        :param input_mask: shape (batch_size, input_seq_len)
        :param length_mask: shape (input_seq_len * self.maximum_fertility)
        :return: a tuple of tensors, both of shape (batch_size,k).
        The first element in the tuple are the log probabilities of the length
        The second element in the tuple is the corresponding length
        """
        input_seq_len = inputs_q.shape[1]
        assert k > 0

        predicted_fertilities = self.compute_fertilities(inputs_q,
                                                         input_mask)  # shape (batch_size, input_seq_len, max fertility)
        if length_mask is not None:
            cumul_sum = sum_of_rand_vars(predicted_fertilities, self.maximum_fertility * input_seq_len)  # shape (batch_size, max length)
            if self.max_length is not None:
                cumul_sum -= 100_000 * (torch.arange(self.maximum_fertility * input_seq_len) > self.max_length)
            cumul_sum += -100_000 * ~length_mask.unsqueeze(0)
        else:
            cumul_sum = sum_of_rand_vars(predicted_fertilities, self.max_length if self.max_length is not None else self.maximum_fertility * input_seq_len) #shape (batch_size, max length)

        cumul_sum[:, 0] = -1.0 #ensure that length of 0 is never chosen.
        values, indices = torch.topk(cumul_sum, k=k, dim=-1)
        return torch.log(values), indices

    def forward(self, inputs_q: torch.Tensor, input_mask: torch.Tensor, target_mask: Optional[torch.Tensor] = None,
                inputs_v: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Computes a marginal monotonic alignment based on a fertility step.
        :param inputs_v: (batch_size, input_seq_len, dim1)
        :param input_mask: bool tensor of shape (batch_size, input_seq_len), where 0 indicates padding.
        :param inputs_q: shape (batch_size, input_seq_len, dim2)
        :param target_mask: (batch_size, output_seq_len) if given.
        :return: a dictionary containing at least
         - output of shape (batch_size, input_seq_len, dim1) if inputs_v provided, else (batch_size, input_seq_len, dim2)
         - log_likelihood of shape (batch_size,) if target_lenghts are provided
        """
        ret = dict()
        if target_mask is None:
            log_prob, target_lengths = self.top_k_lengths(inputs_q, input_mask, 1)
            # predicted_fertilities = self.compute_fertilities(inputs_q, input_mask) #shape (batch_size, input_seq_len, max fertility)
            # predicted_fertilities = torch.argmax(predicted_fertilities, dim=2) #shape (batch_size, input_seq_len)
            # target_lengths = torch.sum(predicted_fertilities, dim=1) #shape (batch_size)


            ret["target_lengths"] = target_lengths
            ret["target_mask"] = get_mask_from_sequence_lengths(target_lengths, int(torch.max(target_lengths).detach().numpy()))

        if inputs_v is None:
            inputs_v = inputs_q

        d = self.compute_marginal_alignment(inputs_q, input_mask, target_mask) #(batch_size, input_seq_len, target_len)
        # print(marginal_alignment[0])
        # print(d["marginal_alignment"].dtype, inputs_v.dtype)
        ret["marginal_alignment"] = d["marginal_alignment"].transpose(1, 2)
        ret["marginal_positions"] = d["marginal_positions"].transpose(1, 2)
        ret["output"] = torch.bmm(ret["marginal_alignment"], inputs_v)

        if self.positional_dim > 0:
            positional_parameters = self.positional_embedding.expand(inputs_v.shape[0], self.maximum_fertility+1, self.positional_dim) #shape (batch_size, m, d)
            marginal_position = ret["marginal_positions"] #shape (batch_size, l, m)
            positional_info = torch.bmm(marginal_position, positional_parameters)
            if self.positional_mode == "sum":
                ret["output"] += positional_info
            elif self.positional_mode == "cat":
                ret["output"] = torch.cat([ret["output"], positional_info], dim=-1)
            else:
                raise NotImplementedError()
            # print(marginal_position.shape, positional_parameters.shape)


        ret["log_likelihood"] = d["log_likelihood"]

        return ret


@FertilityModel.register("joint_fertility")
class JointFertilityStep(FertilityModel):

    def __init__(self, maximum_fertility: int, input_dim: int, positional_dim: int = 0, positional_mode: str = "sum",
                 mlp: Optional[FeedForward] = None, sinusoidal: bool = False,
                 gumbel_temperature: Optional[float] = None,
                 hard: bool = False,
                 temperature: float = 1.0,
                max_length: Optional[int] = None
                 ):
        super(JointFertilityStep, self).__init__(maximum_fertility, input_dim, positional_dim,
                                                 positional_mode, sinusoidal, max_length)
        self.mlp = mlp
        self.gumbel_temperature = gumbel_temperature
        self.hard = hard
        self.temperature = temperature
        if self.mlp:
            self.fertility_layer = torch.nn.Linear(self.mlp.get_output_dim(), self.maximum_fertility + 1)
        else:
            self.fertility_layer = torch.nn.Linear(input_dim,
                                                   self.maximum_fertility + 1)  # +1 because 0 is also a valid fertility.

    def compute_fertilities(self, inputs: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        if self.mlp:
            inputs = self.mlp(inputs)

        fertilities_logits = self.fertility_layer(inputs) / self.temperature  # (batch_size, input_seq_len, m)
        # Create mask to set fertility to 0 for padded elements
        all_mass_on_zero = torch.ones_like(fertilities_logits) * ~input_mask.unsqueeze(2)  # 1 if not padded, else 0
        all_mass_on_zero[:, :, 0] = 0.0  # 1 if not padded or we are in padding but look at P(F_i = 0)

        if self.gumbel_temperature is not None and self.training:
            fertilities = F.gumbel_softmax(-10_000 * all_mass_on_zero + fertilities_logits, tau=self.gumbel_temperature,
                                           hard=self.hard,
                                           dim=2)
        else:
            fertilities = torch.softmax(-10_000 * all_mass_on_zero + fertilities_logits, dim=2)

        return fertilities

    def compute_marginal_alignment(self, inputs: torch.Tensor, input_mask: torch.Tensor, target_mask: torch.Tensor) -> \
    Dict[str, torch.Tensor]:
        batch_size, input_seq_len = input_mask.shape

        assert inputs.shape[
                   -1] == self.input_dim, "input dimensionality must match dimensionality for predicting the fertility"

        fertilities = self.compute_fertilities(inputs, input_mask)

        # print("Fertilities", fertilities)

        target_lengths = get_lengths_from_binary_sequence_mask(target_mask)

        # max_l = int(torch.max(target_mask).cpu().numpy()) + 1
        max_l = target_mask.shape[1] + 1
        forward_sum = cumulative_sum(fertilities,
                                     max_l)  # shape (batch_size, input_seq_len, max_l) where forward_sum[b, i, n] is P(F_0 + ... F+i = n) for that batch element.
        backward_sum = torch.flip(cumulative_sum(torch.flip(fertilities, [1]), max_l), [
            1])  # shape (batch_size, input_seq_len, max_l) where forward_sum[b, i, n] is P(F_n + ... F+_{max_l} = n) for that batch element.

        # joint_alignment = fertility_layer_joint(forward_sum.double(), backward_sum.double(), fertilities, target_lengths)

        joint_alignment = fertility_layer_joint(forward_sum.double(), backward_sum.double(), fertilities, target_lengths)

        # torch.autograd.gradcheck(fertility_layer_joint,
        #                          (forward_sum.double(), backward_sum.double(), fertilities.double(), target_lengths))
        # torch.autograd.gradcheck(better_fertility_layer_joint,
        #                          (forward_sum.double(), backward_sum.double(), fertilities.double(), target_lengths))

        # joint_alignment_2 = better_fertility_layer_joint(forward_sum.double(), backward_sum.double(), fertilities, target_lengths)
        # assert torch.allclose(joint_alignment, joint_alignment_2)
        # assert torch.all(torch.square(joint_alignment - joint_alignment_2) < 0.00001)
        #shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)
        
        return {"joint_alignment": joint_alignment,
                "log_likelihood": torch.log(backward_sum[range(batch_size), 0, target_lengths]),
                "fertilities": fertilities}


    def forward(self, inputs_q: torch.Tensor, input_mask: torch.Tensor, target_mask: Optional[torch.Tensor] = None,
                inputs_v: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Computes a marginal monotonic alignment based on a fertility step.
        :param inputs_v: (batch_size, input_seq_len, dim1)
        :param input_mask: bool tensor of shape (batch_size, input_seq_len), where 0 indicates padding.
        :param inputs_q: shape (batch_size, input_seq_len, dim2)
        :param target_mask: (batch_size, output_seq_len) if given.
        :return: a dictionary containing at least
         - output of shape (batch_size, input_seq_len, dim1) if inputs_v provided, else (batch_size, input_seq_len, dim2)
         - log_likelihood of shape (batch_size,) if target_lenghts are provided
        """
        ret = dict()
        if target_mask is None:
            log_prob, target_lengths = self.top_k_lengths(inputs_q, input_mask, 1)
            # predicted_fertilities = self.compute_fertilities(inputs_q, input_mask) #shape (batch_size, input_seq_len, max fertility)
            # predicted_fertilities = torch.argmax(predicted_fertilities, dim=2) #shape (batch_size, input_seq_len)
            # target_lengths = torch.sum(predicted_fertilities, dim=1) #shape (batch_size)

            ret["target_lengths"] = target_lengths
            ret["target_mask"] = get_mask_from_sequence_lengths(target_lengths, int(torch.max(target_lengths).detach().numpy()))

        if inputs_v is None:
            inputs_v = inputs_q

        d = self.compute_marginal_alignment(inputs_q, input_mask, target_mask) #(batch_size, input_seq_len, target_len)
        # print(marginal_alignment[0])
        # print(d["marginal_alignment"].dtype, inputs_v.dtype)
        joint_alignment = d["joint_alignment"] #shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)
        marginal_alignment = joint_alignment.sum(dim=-1) #shape (batch_size, input_seq_len, output_seq_len)
        marginal_alignment = marginal_alignment.transpose(1, 2)

        ret["output"] = torch.bmm(marginal_alignment.float(), inputs_v)
        ret["joint_alignment"] = joint_alignment
        ret["marginal_alignment"] = marginal_alignment
        ret["fertilities"] = d["fertilities"]

        if self.positional_dim > 0:
            positional_parameters = self.positional_embedding.expand(inputs_v.shape[0], self.maximum_fertility+1, self.positional_dim) #shape (batch_size, m, d)
            marginal_position = joint_alignment.sum(dim=1)  # shape (batch_size, output_seq_len, max fertility + 1)
            positional_info = torch.bmm(marginal_position.float(), positional_parameters)
            if self.positional_mode == "sum":
                ret["output"] += positional_info
            elif self.positional_mode == "cat":
                ret["output"] = torch.cat([ret["output"], positional_info], dim=-1)
            else:
                raise NotImplementedError()
            # print(marginal_position.shape, positional_parameters.shape)


        ret["log_likelihood"] = d["log_likelihood"]

        return ret