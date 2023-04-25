from typing import Dict, Optional, List, Tuple

import numpy as np
import torch
from allennlp.common import Lazy
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, ConditionalRandomField, FeedForward
from allennlp.nn import util, Activation
from allennlp.training.metrics import SequenceAccuracy, CategoricalAccuracy

from allennlp.nn.util import sequence_cross_entropy_with_logits, get_lengths_from_binary_sequence_mask, \
    get_mask_from_sequence_lengths
from torch.nn import LSTMCell

from fertility.conv_utils import my_sequence_cross_entropy_with_logits
from fertility.eval.alignment_acc import AlignmentAccTracker, SimpleAccTracker
from fertility.eval.identity_counter import IdentityCounter, EntropyCounter
from fertility.eval.lev import LevenstheinMetric, MyMetric
from fertility.fertility_model import FertilityModel
from fertility.reordering.lstm import SpanRepresentation
from fertility.reordering.permutation import BinarizableTree
from fertility.scheduler import RateScheduler
from fertility.translation_model import LexicalTranslationModel, LSTMTranslationModel, TranslationModel
from fertility.utils import make_same_length, viterbi_to_array, dump_tensor_dict

import torch.nn.functional as F

import random


@Model.register("discrete_autoregressive")
class DiscreteutoregressiveTransduction(Model):

    def __init__(self, vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 reorder_encoder: Seq2SeqEncoder,
                 fertility_encoder: Seq2SeqEncoder,
                 fertility_model: FertilityModel,
                 embedding_dim: int,
                 mlp: Optional[FeedForward] = None,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 reorder_before_fertility: bool = False,
                 reorder_after_fertility: bool = False,
                 target_namespace: str = "target_tokens",
                 compute_loss_during_dev: bool = True,
                 decoder_dropout: float = 0.1,
                 search_top_k: int = 1,
                 pretrain_epochs: int = 0,
                 length_loss_scheduler: Optional[RateScheduler] = None,
                 drop_grad_rate: float = 0.0,
                 gumbel_temperature: Optional[float] = None,
                 alignment_confidence_thresh: float = 0.8,
                 alignment_loss_weight: float = 0.05,
                 skip_connection: bool = True,
                 source_project_dim: Optional[int] = None,
                 rho: Optional[float] = None,
                 metrics: Optional[List[MyMetric]] = None,
                 pseudo_temperature: float=1
                 ):
        super().__init__(vocab)
        self.reorder_before_fertility = reorder_before_fertility
        self.reorder_after_fertility = reorder_after_fertility
        self.source_embedder = source_embedder
        self.reorder_encoder = reorder_encoder
        self.fertility_encoder = fertility_encoder
        self.seq_acc = SequenceAccuracy()
        self.seq_acc_wo_length = SequenceAccuracy()
        self.tok_acc = CategoricalAccuracy()
        self.length_acc = SimpleAccTracker()
        #self.tok_acc_wo_length = CategoricalAccuracy()
        self.alignment_acc = AlignmentAccTracker()
        self.compute_loss_during_dev = compute_loss_during_dev

        self.metrics = metrics or []

        # self.rho = torch.nn.Parameter(torch.tensor([float(rho)]), requires_grad=True)
        self.rho = rho

        self.pretrain_epochs = pretrain_epochs

        self.length_loss_scheduler = length_loss_scheduler

        self.levenshtein_distance = LevenstheinMetric()
        self.target_namespace = target_namespace
        self.k = search_top_k

        self.fertility_model = fertility_model

        self.encoder = encoder

        self.source_project = None if source_project_dim is None else torch.nn.Linear(source_embedder.get_output_dim(), source_project_dim)

        self.skip_connection = skip_connection
        if self.skip_connection:
            assert self.reorder_encoder.get_input_dim() == self.reorder_encoder.get_output_dim()

        self.drop_grad_rate = drop_grad_rate

        self.identity_counter = IdentityCounter(0.01)
        self.identity_counter2 = IdentityCounter(0.3)
        self.perm_entropy = EntropyCounter()

        self.alignment_confidence_thresh = alignment_confidence_thresh
        self.alignment_loss_weight = alignment_loss_weight

        if self.encoder is None:
            self.dim_after_encoder = self.source_embedder.get_output_dim() if source_project_dim is None else source_project_dim
        else:
            self.dim_after_encoder = self.encoder.get_output_dim()

        self.decoder_hidden_dim = self.dim_after_encoder
        # self.lstm = LSTMCell(self.dim_after_encoder + embedding_dim, self.dim_after_encoder)
        self.lstm = LSTMCell(embedding_dim, self.dim_after_encoder)
        self.decoder_dropout = decoder_dropout

        self.fertility_positional_embedding = torch.nn.Parameter(
            torch.randn([1, 1, self.fertility_model.maximum_fertility+1, self.dim_after_encoder]),
            requires_grad=True)

        self.embedding_weight = torch.nn.Embedding(self.vocab.get_vocab_size(self.target_namespace), embedding_dim)

        assert not (reorder_before_fertility and reorder_after_fertility)

        dim_after_fertility = self.dim_after_encoder + (
            self.fertility_model.positional_dim if self.fertility_model.positional_mode == "cat" else 0)

        if self.reorder_before_fertility:
            self.reorder_module_before = BinarizableTree("cpu", self.reorder_encoder.get_output_dim(),
                                                         use_map_decode=False, wcfg=True,
                                                         gumbel_temperature=gumbel_temperature,
                                                         pseudo_temperature=pseudo_temperature)

            self._bos_for_reorder_before = torch.nn.Parameter(
                torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))
            self._eos_for_reorder_before = torch.nn.Parameter(
                torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))

        if self.reorder_after_fertility:
            self.reorder_module_after = BinarizableTree("cpu", self.reorder_encoder.get_output_dim(),
                                                        use_map_decode=False, wcfg=True,
                                                        gumbel_temperature=gumbel_temperature,
                                                        pseudo_temperature=pseudo_temperature)

            self._bos_for_reorder_after = torch.nn.Parameter(torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))
            self._eos_for_reorder_after = torch.nn.Parameter(torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))

        self.mlp = mlp

        if self.mlp is not None:
            assert self.mlp.get_input_dim() == self.decoder_hidden_dim

        # self.output_layer = torch.nn.Linear(self.decoder_hidden_dim, self.vocab.get_vocab_size(self.target_namespace))
        self.output_layer = torch.nn.Linear(self.mlp.get_output_dim(), (self.fertility_model.maximum_fertility + 1) * self.vocab.get_vocab_size(self.target_namespace))

    def compute_logits(self, encoder_outputs: torch.Tensor, embedded_source: torch.Tensor, source_mask: torch.Tensor,
                       target_mask: Optional[torch.Tensor] = None, target_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute logits for a given input. If target_mask is given (and thereby the target lengths), we use that to compute the logits.
        Otherwise, the fertility model guesses the most likely target lengths.

        Returns the logits for the output sequence and a dictionary with the log likelihood of the target lengths
        and information about the fertility and permutation matrices.
        :param source_mask:
        :param encoder_outputs: shape (batch_size, input_seq_len, dim)
        :param target_mask:
        :return:
        """
        if hasattr(self, "epoch") and random.random() < self.drop_grad_rate:
            self.lstm.requires_grad_(False)
        else:
            self.lstm.requires_grad_(True)

        d = dict()
        perm_matrix = None
        if self.reorder_before_fertility:
            encoded_for_reordering = self.reorder_encoder(embedded_source, source_mask)
            # We need to add a BOS and EOS token for the extraction of features
            eos = self._eos_for_reorder_before.expand(encoded_for_reordering.shape[0], 1,
                                                     encoded_for_reordering.shape[-1])
            bos = self._bos_for_reorder_before.expand(encoded_for_reordering.shape[0], 1,
                                                     encoded_for_reordering.shape[-1])

            padded_encoder_outputs = torch.cat([bos, encoded_for_reordering, eos], dim=1)
            span_rep = SpanRepresentation(padded_encoder_outputs,
                                          get_lengths_from_binary_sequence_mask(source_mask) + 2)  # +2 for BOS and EOS
            perm_matrix, _ = self.reorder_module_before(span_rep)  # shape (batch_size, source len, source len)
            d["reorder_before_fertility"] = perm_matrix
            output_after_reordering = torch.bmm(perm_matrix, embedded_source)

            encoded_for_fertility = self.fertility_encoder(output_after_reordering, source_mask)
            if self.skip_connection:
                encoded_for_fertility += output_after_reordering

        else:
            encoded_for_fertility = self.fertility_encoder(embedded_source, source_mask)
        # if self.skip_connection:
        # encoded_for_fertility += embedded_source

        d.update(self.fertility_model(encoded_for_fertility, source_mask, target_mask, inputs_v=embedded_source))

        # d.update(self.fertility_model(embedded_source, source_mask, target_mask, inputs_v=embedded_source))

        if target_mask is None:
            target_mask = d["target_mask"]

        pred_encoder_outputs = d[
            "output"]  # shape (batch_size, target_len, encoder dim + positional embedding - depending on the options)

        if self.reorder_after_fertility:
            encoded_for_reordering = self.reorder_encoder(pred_encoder_outputs, target_mask)
            if self.skip_connection:
                # seems to be important to avoid learning trivial re-ordering strategies.
                encoded_for_reordering += pred_encoder_outputs  # skip connection!
            # We need to add a BOS and EOS token for the extraction of features
            eos = self._eos_for_reorder_after.expand(encoded_for_reordering.shape[0], 1,
                                                     encoded_for_reordering.shape[-1])
            bos = self._bos_for_reorder_after.expand(encoded_for_reordering.shape[0], 1,
                                                     encoded_for_reordering.shape[-1])

            padded_encoder_outputs = torch.cat([bos, encoded_for_reordering, eos], dim=1)
            span_rep = SpanRepresentation(padded_encoder_outputs,
                                          get_lengths_from_binary_sequence_mask(target_mask) + 2)  # +2 for BOS and EOS
            perm_matrix, _ = self.reorder_module_after(span_rep)  # shape (batch_size, target len, target len)
            d["reorder_after_fertility"] = perm_matrix

            # pred_encoder_outputs = torch.bmm(perm_matrix, pred_encoder_outputs)

        joint_alignment = d["joint_alignment"]  # shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)

        if self.reorder_after_fertility:
            alignment_after_fertility_and_reorder = torch.einsum("box, bixk  -> biok", perm_matrix.double(),
                                                                 joint_alignment)  # shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)
            # i -> x -> o
        elif self.reorder_before_fertility:
            alignment_after_fertility_and_reorder = torch.einsum("bxi, bxok -> biok", perm_matrix.double(), joint_alignment)
            # i -> x -> o
            # shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)
        else:
            # No re-ordering
            alignment_after_fertility_and_reorder = joint_alignment #.sum(dim=-1)

        d["alignment_after_fertility_and_reorder"] = alignment_after_fertility_and_reorder

        decoder_hidden = torch.zeros((encoder_outputs.shape[0], self.decoder_hidden_dim), device=encoder_outputs.device)
        decoder_context = torch.zeros_like(decoder_hidden)
        dropout_mask = F.dropout(torch.ones_like(decoder_hidden), self.decoder_dropout, self.training, inplace=False)

        max_output_seq_len = target_mask.shape[1]
        output_probs = []
        for o in range(max_output_seq_len):
            # old:
            # joint_repr = torch.sigmoid(self.gamma) * decoder_hidden.unsqueeze(1).unsqueeze(1) + encoder_outputs.unsqueeze(2) + self.fertility_positional_embedding  #shape (batch_size, input_seq_len, maximum fertility + 1, some dim)


            # joint_repr = torch.cat() decoder_hidden.unsqueeze(1).unsqueeze(1) + encoder_outputs.unsqueeze(2) + self.fertility_positional_embedding  #shape (batch_size, input_seq_len, maximum fertility + 1, some dim)
            # joint_probs = torch.softmax(self.output_layer(self.mlp(joint_repr)).double(), dim=-1) #shape (batch_size, input_seq_len, max fertility + 1, vocab)

            # new:
            # joint_repr = torch.sigmoid(self.gamma) * decoder_hidden.unsqueeze(1) + encoder_outputs # shape (batch_size, input_seq_len, some dim)
            joint_repr = decoder_hidden.unsqueeze(1) + encoder_outputs # shape (batch_size, input_seq_len, some dim)
            joint_repr = self.output_layer(self.mlp(joint_repr)).double()
            joint_logits = joint_repr.reshape([encoder_outputs.shape[0], encoder_outputs.shape[1], self.fertility_model.maximum_fertility + 1,
                           self.vocab.get_vocab_size(self.target_namespace)])
            joint_probs = torch.softmax(joint_logits,
                                        dim=-1)  # shape (batch_size, input_seq_len, max fertility + 1, vocab)

            current_alignment = alignment_after_fertility_and_reorder[:, :, o, :]
            output_prob = torch.einsum("bikv, bik -> bv", joint_probs, current_alignment)

            output_probs.append(output_prob)

            if target_tokens is not None:
                embedded_last_output = self.embedding_weight(target_tokens[:, o])

            else:
                # Greedy
                embedded_last_output = self.embedding_weight(torch.argmax(output_prob, dim=-1))

            decoder_hidden, decoder_context = self.lstm(
                embedded_last_output, (decoder_hidden, decoder_context)
            )
            decoder_hidden *= dropout_mask

        return torch.log(torch.stack(output_probs, dim=1) + 1e-10).contiguous(), d

    def most_likely_sequence_for_length(self, logits: torch.Tensor, target_mask: torch.Tensor,
                                        proper_log_probs: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the most likely output seqeuence given the logits.
        :param target_mask: shape (batch_size, seq_len)
        :param logits: shape (batch_size, seq_len, vocab size)
        :return: a tensor of shape (batch_size, seq_len,) with the most likely representation
        and a tensor of shape (batch_size,) with its coresponding log probabilty.
        """
        top_1_predictions_with_length = torch.argmax(logits, dim=2)  # shape (batch_size, input_seq_length)
        top_1_log_probs = -my_sequence_cross_entropy_with_logits(logits, top_1_predictions_with_length, target_mask,
                                                                 average=None).sum(dim=-1)
        # shape (batch_size,)

        return top_1_predictions_with_length, top_1_log_probs


    def top_k_lengths(self, embedded_source: torch.Tensor, source_mask: torch.Tensor, k:int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute top k lengths according to fertility model.
        :param embedded_source:
        :param source_mask:
        :param k:
        :return:
        """
        # TODO: this is DANGEROUS if we need to re-order first before computing the fertility!
        # It relies for the prediction of total length to be order invariant (which it's not if it uses an LSTM beforehand)!
        # assert not self.reorder_before_fertility
        if self.reorder_before_fertility:
            encoded_for_reordering = self.reorder_encoder(embedded_source, source_mask)
            # We need to add a BOS and EOS token for the extraction of features
            eos = self._eos_for_reorder_before.expand(encoded_for_reordering.shape[0], 1,
                                                      encoded_for_reordering.shape[-1])
            bos = self._bos_for_reorder_before.expand(encoded_for_reordering.shape[0], 1,
                                                      encoded_for_reordering.shape[-1])

            padded_encoder_outputs = torch.cat([bos, encoded_for_reordering, eos], dim=1)
            span_rep = SpanRepresentation(padded_encoder_outputs,
                                          get_lengths_from_binary_sequence_mask(source_mask) + 2)  # +2 for BOS and EOS
            perm_matrix, _ = self.reorder_module_before(span_rep)  # shape (batch_size, source len, source len)
            output_after_reordering = torch.bmm(perm_matrix, embedded_source)

            encoded_for_fertility = self.fertility_encoder(output_after_reordering, source_mask)
            if self.skip_connection:
                encoded_for_fertility += output_after_reordering

        else:
            encoded_for_fertility = self.fertility_encoder(embedded_source, source_mask)

        top_k_log_probs, top_k_lengths = self.fertility_model.top_k_lengths(encoded_for_fertility, source_mask, k)
        return top_k_log_probs, top_k_lengths


    def most_likely_sequence(self, encoder_outputs, embedded_source, source_mask, k: int) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Finds the k most likely lengths and finds the most likely overall sequence for all of the k lengths.
        Returns the most likely sequence and the respective length
        with shapes (batch_size, maximum seq length) and (batch_size,) respectively.
        :return:
        """
        top_k_log_probs, top_k_lengths = self.top_k_lengths(embedded_source, source_mask, k)

        max_l = int(torch.max(top_k_lengths).detach().cpu().numpy())

        best_sequences = torch.zeros((encoder_outputs.shape[0], max_l), dtype=torch.long, device=encoder_outputs.device)
        best_lengths = torch.zeros(encoder_outputs.shape[0], dtype=torch.long, device=encoder_outputs.device)
        best_log_prob = torch.zeros(encoder_outputs.shape[0], device=encoder_outputs.device) - 1_000_000

        for i in range(k):
            # shape (batch_size)
            curr_seq_length = top_k_lengths[:, i]
            curr_seq_log_prob = top_k_log_probs[:, i]

            target_mask = get_mask_from_sequence_lengths(curr_seq_length, max_l)
            logits, debug_info = self.compute_logits(encoder_outputs, embedded_source, source_mask, target_mask)
            most_likely_seq, log_prob_seq = self.most_likely_sequence_for_length(logits, target_mask)
            # shape (batch, max_l) and (batch)

            current_log_prob = curr_seq_log_prob + log_prob_seq

            current_length_is_better = current_log_prob >= best_log_prob  # shape (batch_size,)
            best_log_prob = torch.maximum(best_log_prob, current_log_prob)

            best_lengths = current_length_is_better * curr_seq_length + ~current_length_is_better * best_lengths

            current_length_is_better = current_length_is_better.unsqueeze(1)  # shape (batch, 1)
            best_sequences = current_length_is_better * most_likely_seq + ~current_length_is_better * best_sequences

        if k == 1:
            del debug_info["output"]
            return best_sequences, best_lengths, debug_info

        return best_sequences, best_lengths, dict()

    def pretrain_loss(self, predicted_alignment: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor,
                      alignment: torch.Tensor):
        """
        :param predicted_alignment: shape (batch, input seq len, output seq len)
        :return:
        """
        loss = (torch.log(predicted_alignment + 1e-30) * (alignment >= self.alignment_confidence_thresh)).sum(
            dim=[1, 2])  # shape (batch_size,)

        return loss

    def forward(self, source_tokens: TextFieldTensors,
                metadata: List[Dict],
                target_tokens: Optional[TextFieldTensors] = None,
                alignment: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_source = self.source_embedder(source_tokens)
        if self.source_project is not None:
            embedded_source = self.source_project(embedded_source)

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        if self.encoder:
            encoder_outputs = self.encoder(embedded_source, source_mask)
            if self.rho is not None:
                encoder_outputs = self.rho*encoder_outputs + embedded_source
        else:
            encoder_outputs = embedded_source

        readable_source = [m["source_tokens"] for m in metadata]

        ret = {"source_lengths": get_lengths_from_binary_sequence_mask(source_mask),
               "readable_source": readable_source}

        if target_tokens:
            targets = target_tokens["tokens"]["tokens"]  # shape (batch_size, output seq length)
            target_mask = util.get_text_field_mask(target_tokens)  # shape (batch_size, output seq length)
            target_lengths = get_lengths_from_binary_sequence_mask(target_mask)
            ret["targets"] = targets
            ret["gold_target_lengths"] = get_lengths_from_binary_sequence_mask(target_mask)

            readable_target = [m["target_tokens"] for m in metadata]
            ret["readable_targets"] = readable_target

            # assert torch.all(self.oov_id != targets), "One of the target tokens was the OOV symbol. " \
            #                                           "The correctness of the copy mechanism relies on OOV to never occur."

        if not self.training:
            # Make predictions
            predictions, predicted_lengths, debug_info = self.most_likely_sequence(encoder_outputs, embedded_source,
                                                                                   source_mask, self.k)
            ret["predicted_target_lengths"] = predicted_lengths
            ret["predictions"] = predictions
            ret.update(debug_info)
            readable_prediction = self.make_readable(predictions, predicted_lengths, self.target_namespace)

            if target_tokens:
                self.levenshtein_distance.add_instances(readable_prediction,
                    readable_target)

                for m in self.metrics:
                    m.add_instances(readable_prediction, readable_target)

                # print("Preds:", pred[0], gold[0], gold_mask[0])
                # self.tok_acc_wo_length(pred, gold, gold_mask)

        if target_tokens and (self.training or self.compute_loss_during_dev):
            loss = torch.zeros(1, device=encoder_outputs.device)
            output_logits, d = self.compute_logits(encoder_outputs, embedded_source, source_mask, target_mask, targets)
            for k, v in d.items():
                if k != "output":
                    ret[k + "_given_length"] = v

            if "reorder_after_fertility" in d:
                self.identity_counter.add_matrix(d["reorder_after_fertility"], target_lengths)
                self.identity_counter2.add_matrix(d["reorder_after_fertility"], target_lengths)
                self.perm_entropy.add_matrix(d["reorder_after_fertility"], target_mask)
            elif "reorder_before_fertility" in d:
                source_lengths = source_mask.sum(dim=-1) #shape (batch_size,)
                self.identity_counter.add_matrix(d["reorder_before_fertility"], source_lengths)
                self.identity_counter2.add_matrix(d["reorder_before_fertility"], source_lengths)
                self.perm_entropy.add_matrix(d["reorder_before_fertility"], source_mask)

            total_alignment = d["alignment_after_fertility_and_reorder"].sum(dim=-1) # shape (batch_size, input_seq_len, output_seq_len)

            if alignment is not None:
                self.alignment_acc(total_alignment, alignment,
                                   target_mask * torch.any(alignment >= self.alignment_confidence_thresh, dim=1))

            if hasattr(self, "epoch") and self.epoch < self.pretrain_epochs and alignment is not None:
                # smoothed_alignment = (alignment + 1e-20) ** self.smooth_alignment
                # smoothed_alignment = smoothed_alignment / smoothed_alignment.sum(dim=-1, keepdims=True)

                alignment_loss = self.alignment_loss_weight * self.pretrain_loss(total_alignment, source_mask, target_mask
                                                                         , alignment)
                loss += -alignment_loss.sum()

            length_logl = d["log_likelihood"]

            if self.training and self.length_loss_scheduler:
                if not hasattr(self, "epoch"):
                    raise ValueError(
                        "Model was given a length loss scheduler but we don't know the epoch, add a track_epoch_callback")
                r = self.length_loss_scheduler.get_rate(self.epoch)
                loss += -r * length_logl.sum()
            else:
                loss += - length_logl.sum()

            top_1_predictions, _ = self.most_likely_sequence_for_length(output_logits, target_mask,
                                                                        proper_log_probs=False)

            ret["predictions_given_length"] = top_1_predictions

            self.seq_acc(top_1_predictions.unsqueeze(1), targets, target_mask)
            self.tok_acc(output_logits, targets, target_mask)

            _, most_likely_lengths = self.top_k_lengths(embedded_source, source_mask, 1) #shape (batch_size, 1)
            self.length_acc(most_likely_lengths.squeeze(1), target_lengths)

            loss += my_sequence_cross_entropy_with_logits(output_logits, targets, target_mask)

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                data = {"loss": loss, "length_logl": length_logl, "total_alignment": total_alignment, "d": d,
                        "alignment": alignment}
                dump_tensor_dict(data, "crashed_discrete_nat.pickle")
                raise ValueError("Nan or inf loss")

            ret["loss"] = loss

        return ret

    def make_readable(self, batch_of_tokens: torch.Tensor, lengths: torch.Tensor, namespace: str) -> List[List[str]]:
        r = []
        for batch, length_of_seq in zip(batch_of_tokens.cpu().numpy(), lengths.cpu().numpy()):
            r.append([self.vocab.get_token_from_index(tok, namespace=namespace) for tok in batch[:length_of_seq]])
        return r

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        if "predictions" in output_dict:
            output_dict["readable_predictions"] = self.make_readable(output_dict["predictions"],
                                                                     output_dict["predicted_target_lengths"],
                                                                     self.target_namespace)

        if "predictions_given_length" in output_dict:
            output_dict["readable_predictions_given_length"] = self.make_readable(
                output_dict["predictions_given_length"], output_dict["gold_target_lengths"], self.target_namespace)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = {"seq_acc_given_length": self.seq_acc.get_metric(reset)["accuracy"],
             "tok_acc": self.tok_acc.get_metric(reset)}
        d.update(self.identity_counter.get_metrics(reset))
        d.update(self.perm_entropy.get_metrics(reset))
        d["alignment_acc"] = self.alignment_acc.get_metric(reset)
        d["length_acc"] = self.length_acc.get_metric(reset)
        # d["gamma"] = float(torch.sigmoid(self.gamma.detach().cpu()).numpy())
        d["frac_identities_thresh_0.3"] = self.identity_counter2.get_metrics(reset)["frac_identities"]
        if not self.training:
            d.update(self.levenshtein_distance.get_metric(reset))

        for m in self.metrics:
            d.update(m.get_metric(reset))
        return d