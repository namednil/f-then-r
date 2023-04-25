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

from fertility.conv_utils import my_sequence_cross_entropy_with_logits
from fertility.decoding.decoding_grammar import DecodingGrammar
from fertility.eval.alignment_acc import AlignmentAccTracker, SimpleAccTracker
from fertility.eval.identity_counter import IdentityCounter, EntropyCounter
from fertility.eval.lev import LevenstheinMetric, MyMetric
from fertility.fertility_model import FertilityModel
from fertility.reordering.lstm import SpanRepresentation
from fertility.reordering.permutation import BinarizableTree
from fertility.scheduler import RateScheduler
from fertility.test_mode import TestMode
from fertility.translation_model import LexicalTranslationModel, LSTMTranslationModel, TranslationModel
from fertility.utils import make_same_length, viterbi_to_array, dump_tensor_dict

import torch.nn.functional as F

import random

COPY_SYMBOL = "@COPY@"


@Model.register("discrete_copy_nat")
class DiscreteNonautoregressiveTransductionWithCopying(Model):

    def __init__(self, vocab: Vocabulary,
                 source_embedder: TextFieldEmbedder,
                 reorder_encoder: Seq2SeqEncoder,
                 fertility_encoder: Seq2SeqEncoder,
                 fertility_model: FertilityModel,
                 mlp: Optional[FeedForward] = None,
                 translation_model: Optional[Lazy[TranslationModel]] = None,
                 encoder: Optional[Seq2SeqEncoder] = None,
                 reorder_before_fertility: bool = False,
                 reorder_after_fertility: bool = False,
                 target_namespace: str = "target_tokens",
                 compute_loss_during_dev: bool = True,
                 use_crf: bool = False,
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
                 label_smoothing: Optional[float] = None,
                 use_nlg_metrics: bool = False,
                 drop_context: float = 0.0,
                 metrics: Optional[List[MyMetric]] = None,
                 use_gelu : bool = False,
                 grammar: Optional[Lazy[DecodingGrammar]] = None,
                 use_map_tree: bool = False,
                 test_search_top_k: int = 1,
                 dev_use_grammar: bool = True,
                 test_use_grammar: bool = True,
                 pseudo_temperature: float=1,
                 closed_vocabulary_grammar: bool = False
                 ):
        super().__init__(vocab)

        self.closed_vocabulary_grammar = closed_vocabulary_grammar
        if closed_vocabulary_grammar and grammar is not None:
            # quickly construct the grammar to be able to extract the terminals
            terminals = sorted(grammar.construct(tok2id=dict()).get_terminals())
            self.vocab.add_tokens_to_namespace(terminals, target_namespace)

        assert COPY_SYMBOL in vocab.get_token_to_index_vocabulary(namespace=target_namespace)

        self.copy_id = vocab.get_token_index(COPY_SYMBOL, namespace=target_namespace)
        self.oov_id = self.vocab.get_token_index(self.vocab._oov_token, namespace=target_namespace)

        self.reorder_before_fertility = reorder_before_fertility
        self.reorder_after_fertility = reorder_after_fertility
        self.source_embedder = source_embedder
        self.reorder_encoder = reorder_encoder
        self.fertility_encoder = fertility_encoder
        self.seq_acc = SequenceAccuracy()
        # self.seq_acc_wo_length = SequenceAccuracy()
        self.tok_acc = CategoricalAccuracy()
        self.length_acc = SimpleAccTracker()
        # self.tok_acc_wo_length = CategoricalAccuracy()
        self.alignment_acc = AlignmentAccTracker()
        self.compute_loss_during_dev = compute_loss_during_dev

        self.label_smoothing = label_smoothing

        self.rho = rho
        self.drop_context = drop_context

        self.metrics = metrics or []

        self.pretrain_epochs = pretrain_epochs

        self.length_loss_scheduler = length_loss_scheduler

        self.levenshtein_distance = LevenstheinMetric()
        self.levenshtein_distance_given_length = LevenstheinMetric()
        self.target_namespace = target_namespace
        self.k = search_top_k

        self.test_k = test_search_top_k
        self.test_use_grammar = test_use_grammar
        self.dev_use_grammar = dev_use_grammar
        self.test_mode = TestMode(metrics, True)


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

        self.dim_after_encoder = self.source_embedder.get_output_dim() if self.encoder is None else self.encoder.get_output_dim()

        if mlp and translation_model:
            raise ConfigurationError("Cannot use mlp (for backwards compatibility) and translation_model at the same time")

        if not mlp and not translation_model:
            raise ConfigurationError("Need either mlp or translation_model")

        if mlp:
            self.translation_model = LexicalTranslationModel(self.vocab, self.fertility_model.maximum_fertility,
                                                             self.target_namespace, mlp)
        else:
            self.translation_model = translation_model.construct(vocab=self.vocab,
                                                                 maximum_fertility=self.fertility_model.maximum_fertility,
                                                                 target_namespace=self.target_namespace)

        if grammar:
            self.grammar = grammar.construct(tok2id=self.vocab.get_token_to_index_vocabulary(self.target_namespace))
            assert not use_crf
        else:
            self.grammar = None

        assert self.translation_model.get_input_dim() == self.dim_after_encoder

        if use_crf:
            self.crf: Optional[ConditionalRandomField] = ConditionalRandomField(
                self.vocab.get_vocab_size(target_namespace))
        else:
            self.crf: Optional[ConditionalRandomField] = None

        assert not (reorder_before_fertility and reorder_after_fertility)

        dim_after_fertility = self.dim_after_encoder + (
            self.fertility_model.positional_dim if self.fertility_model.positional_mode == "cat" else 0)

        if self.reorder_before_fertility:
            self.reorder_module_before = BinarizableTree("cpu", self.reorder_encoder.get_output_dim(),
                                                         use_map_decode=use_map_tree, wcfg=True,
                                                         use_gelu=use_gelu,
                                                         gumbel_temperature=gumbel_temperature,
                                                         pseudo_temperature=pseudo_temperature)

            self._bos_for_reorder_before = torch.nn.Parameter(
                torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))
            self._eos_for_reorder_before = torch.nn.Parameter(
                torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))

        if self.reorder_after_fertility:
            self.reorder_module_after = BinarizableTree("cpu", self.reorder_encoder.get_output_dim(),
                                                        use_map_decode=use_map_tree, wcfg=True,
                                                        use_gelu=use_gelu,
                                                        gumbel_temperature=gumbel_temperature,
                                                        pseudo_temperature=pseudo_temperature)

            self._bos_for_reorder_after = torch.nn.Parameter(torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))
            self._eos_for_reorder_after = torch.nn.Parameter(torch.randn([1, 1, self.reorder_encoder.get_output_dim()]))


        # This is only used for StylePTB for the emphasis tasks.
        self._highlighting_param = torch.nn.Parameter(torch.randn([1, 1,
                                                                   self.source_embedder.get_output_dim()
                                                                   if source_project_dim is None
                                                                   else source_project_dim]), requires_grad=True)

        # if self.mlp is not None:
        # assert self.mlp.get_input_dim() == dim_after_fertility
        # self.output_layer = torch.nn.Linear(mlp.get_output_dim(), self.vocab.get_vocab_size(self.target_namespace))
        # else:
        # self.output_layer = torch.nn.Linear(dim_after_fertility, self.vocab.get_vocab_size(self.target_namespace))

        if use_nlg_metrics:
            from fertility.eval.nlg_eval import NLGMetrics
            self.nlg_metrics = NLGMetrics()
        else:
            self.nlg_metrics = None

    def compute_logits(self, encoder_outputs: torch.Tensor, embedded_source: torch.Tensor, source_mask: torch.Tensor,
                       target_mask: Optional[torch.Tensor] = None,
                       source_to_copy_mask: Optional[torch.Tensor] = None,
                       copyable_inputs: Optional[torch.Tensor] = None,
                       source2vocab: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute logits for a given input. If target_mask is given (and thereby the target lengths), we use that to compute the logits.
        Otherwise, the fertility model guesses the most likely target lengths.

        Returns the logits for the output sequence and a dictionary with the log likelihood of the target lengths
        and information about the fertility and permutation matrices.
        :param source_to_copy_mask: shape (batch_size, input seq len, output seq len)
        :param copyable_inputs: shape (batch_size, input seq len)
        :param source_mask: shape (batch_size, input seq len)
        :param encoder_outputs:shape (batch_size, input seq len, encoder dim)
        :param target_mask: shape (batch_size, output seq len)
        :return:
        """

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

        if self.reorder_before_fertility:
            # In order to get fertilities that are associated with input tokens, we have to undo the soft permutation
            # As our soft permutations approach hard permutations over training, transposing them makes sense to get an
            # "inverse"

            d["fertilities_for_input_tokens"] = torch.bmm(d["reorder_before_fertility"].transpose(1, 2), d["fertilities"]) #shape (batch_size, isl, isl) * (batch_size, isl, max fert.) -> (batch_size, isl, max fert.)

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
            d["fertilities_for_input_tokens"] = d["fertilities"]

        translation_logprobs = self.translation_model(encoder_outputs,
                                                      source_mask, d["fertilities_for_input_tokens"])  # shape (batch_size, input_seq_len, max fertility + 1, vocab size)

        joint_alignment = d["joint_alignment"]  # shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)
        if self.reorder_after_fertility:
            alignment_after_fertility_and_reorder = torch.einsum("box, bixk  -> biok", perm_matrix.double(),
                                                                 joint_alignment)  # shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)
            # i -> x -> o
        elif self.reorder_before_fertility:
            alignment_after_fertility_and_reorder = torch.einsum("bxi, bxok -> biok", perm_matrix.double(),
                                                                 joint_alignment)
            # i -> x -> o
            # shape (batch_size, input_seq_len, output_seq_len, max fertility + 1)
        else:
            # No re-ordering
            alignment_after_fertility_and_reorder = joint_alignment

        translation_probs = translation_logprobs.double().exp()  # shape (batch_size, input_seq_len, max fertility, vocab)

        # Create mask for copy symbol
        copy_part = torch.zeros((1, 1, self.vocab.get_vocab_size(namespace=self.target_namespace)), dtype=torch.bool,
                                device=translation_probs.device) #shape ("batch", "output", vocab), where "batch" and "output" are set to 1.
        copy_part[:, :, self.copy_id] = True

        # Separate probability for copying from outputting proper vocab tokens
        copy_probs = translation_probs[:, :, :, self.copy_id]  # shape (batch_size, input_seq_len, max fertility)
        translation_probs = translation_probs * ~copy_part.unsqueeze(1) #sets probability for generating copy symbol (from vocabulary) to 0

        # TODO: we have to additional events: (correct) copy and incorrect copy (= bin)
        # at training time: move the probability mass of tokens that should not be copied from COPY to the bin
        # if
        # translation_probs[b, i, k, bin] = translation_probs[b, i, k, copy]
        #

        # TODO: we have something in the output that could be generated by copying or by using the usual generation mechanism.
        # -> ignore???

        if source_to_copy_mask is not None:
            # we are at training time, only certain inputs are eligible for copying

            # there might be multiple ways to copy correctly, marginalize over them.
            correct_copy_probs = torch.einsum("bik, biok, bio -> bo", copy_probs,
                                              alignment_after_fertility_and_reorder, source_to_copy_mask)
            # probabilities of copying in an incorrect way
            incorrect_copy_probs = torch.einsum("bik, biok, bio -> bo", copy_probs,
                                                alignment_after_fertility_and_reorder,
                                                (1 - source_to_copy_mask) * source_mask.unsqueeze(2) * target_mask.unsqueeze(1))
            # put the mass of incorrect copying on the OOV symbol, which we know is never used, so always incorrect.
            oov_part = torch.zeros_like(copy_part)
            oov_part[:, :, self.oov_id] = True

            output_probs = copy_part * correct_copy_probs.unsqueeze(2) + \
                           oov_part * incorrect_copy_probs.unsqueeze(2) + \
                           torch.einsum("bikv, biok -> bov", translation_probs, alignment_after_fertility_and_reorder)

        else:

            if source2vocab is not None:
                # We are given a mapping between source tokens and the target vocabulary (it's a closed vocabulary),
                # so we can directly map the copy probability into the target vocabulary.
                # this is useful for grammar-based decoding.
                output_probs = torch.einsum("bik, biok, biv -> bov", copy_probs, alignment_after_fertility_and_reorder, source2vocab) \
                             + torch.einsum("bikv, biok -> bov", translation_probs, alignment_after_fertility_and_reorder)
                # set some dummy values...
                d["copy_backptr"] = torch.zeros_like(target_mask,
                                                     dtype=torch.long)  # shape (batch_size, output seq len)

            else:
                # We don't have a mapping between source tokens and the target vocabulary (because it's an open vocabulary)
                # so we marginalise over all possible ways to copy and assign that probability mass to producing the COPY symbol.
                # and set a pointer to the index in the source what that COPY symbol corresponds to.

                if copyable_inputs is not None:
                    copy_probs = torch.einsum("bik, biok, bi -> bio", copy_probs,
                                              alignment_after_fertility_and_reorder, copyable_inputs)
                else:
                    # we are at test time, every input can copy
                    copy_probs = torch.einsum("bik, biok -> bio", copy_probs,
                                              alignment_after_fertility_and_reorder)
                # probability of aligning a certain input to a certain output token AND deciding to copy.

                # = torch.argmax(copy_probs, dim=1) #shape (batch_size, output_seq_len)
                # -> if we copy at a certain output position, where did this come from?
                # copy_probs = copy_probs.sum(dim=1) #shape (batch_size, output seq len)
                _, d["copy_backptr"] = copy_probs.max(dim=1)  # shape (batch_size, output seq len)
                copy_probs = copy_probs.sum(dim=1)

                output_probs = copy_part * copy_probs.unsqueeze(2) + torch.einsum("bikv, biok -> bov", translation_probs,
                                            alignment_after_fertility_and_reorder)


        # print(output_probs.sum(dim=-1))

        # output_probs_copy = copy_part * copy_probs.unsqueeze(2) #shape (batch_size, output_seq_len, vocab)
        # # -> 0 for everything that isn't copy
        # output_probs = output_probs_copy + torch.einsum("bikv, biok -> bov", translation_probs, alignment_after_fertility_and_reorder)

        d["alignment_after_fertility_and_reorder"] = alignment_after_fertility_and_reorder

        return torch.log(output_probs + 1e-10).contiguous(), d

    def most_likely_sequence_for_length(self, logits: torch.Tensor, target_mask: torch.Tensor,
                                        proper_log_probs: bool = True, use_grammar: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the most likely output seqeuence given the logits.
        :param target_mask: shape (batch_size, seq_len)
        :param logits: shape (batch_size, seq_len, vocab size)
        :return: a tensor of shape (batch_size, seq_len,) with the most likely representation
        and a tensor of shape (batch_size,) with its coresponding log probabilty.
        """
        if self.crf is not None:
            viterbi_tags = self.crf.viterbi_tags(logits, target_mask)
            # For some reason, the function returns a list with ints and floats...
            top_1_predictions_with_length = torch.from_numpy(viterbi_to_array(viterbi_tags)).to(logits.device)
            # we might need to pad, want to ensure that output has the shape of logits with last axis removed.
            top_1_predictions_with_length = F.pad(top_1_predictions_with_length,
                                                  (0, target_mask.shape[1] - top_1_predictions_with_length.shape[1]))
            top_1_log_probs = torch.from_numpy(np.array([score for (_, score) in viterbi_tags])).to(
                logits.device)  # shape (batch_size,)

            if proper_log_probs:
                # need to compute log partition function.
                top_1_log_probs = top_1_log_probs - self.crf._input_likelihood(logits, target_mask)

        else:
            if self.grammar and use_grammar:
                lengths = target_mask.sum(dim=-1)
                return self.grammar.batch_decode(logits, lengths, use_ints=True)
            else:
                top_1_predictions_with_length = torch.argmax(logits, dim=2)  # shape (batch_size, input_seq_length)
                top_1_log_probs = -my_sequence_cross_entropy_with_logits(logits, top_1_predictions_with_length, target_mask,
                                                                         average=None).sum(dim=-1)
            # shape (batch_size,)

        return top_1_predictions_with_length, top_1_log_probs

    def top_k_lengths(self, embedded_source: torch.Tensor, source_mask: torch.Tensor, k: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
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

        if self.grammar:
            max_len = self.fertility_model.maximum_fertility * source_mask.shape[1]
            length_mask = self.grammar.get_derivable_lengths(max_len).to(encoded_for_fertility.device)
            top_k_log_probs, top_k_lengths = self.fertility_model.top_k_lengths(encoded_for_fertility, source_mask, k,
                                                                                length_mask=length_mask)
        else:
            top_k_log_probs, top_k_lengths = self.fertility_model.top_k_lengths(encoded_for_fertility, source_mask, k)
        return top_k_log_probs, top_k_lengths




    def most_likely_sequence(self, encoder_outputs, embedded_source, source_mask, k: int, use_grammar: bool = True, copyable_inputs: Optional[torch.Tensor] = None,
                             source2vocab: Optional[torch.Tensor] = None) -> Tuple[
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
        best_backptr = torch.zeros((encoder_outputs.shape[0], max_l), dtype=torch.long, device=encoder_outputs.device)

        for i in range(k):
            # shape (batch_size)
            curr_seq_length = top_k_lengths[:, i]
            curr_seq_log_prob = top_k_log_probs[:, i]

            target_mask = get_mask_from_sequence_lengths(curr_seq_length, max_l)
            logits, debug_info = self.compute_logits(encoder_outputs, embedded_source, source_mask, target_mask,
                                                     copyable_inputs=copyable_inputs, source2vocab=source2vocab)
            most_likely_seq, log_prob_seq = self.most_likely_sequence_for_length(logits, target_mask, use_grammar=use_grammar)
            # shape (batch, max_l) and (batch)

            current_log_prob = curr_seq_log_prob + log_prob_seq

            current_length_is_better = current_log_prob >= best_log_prob  # shape (batch_size,)
            best_log_prob = torch.maximum(best_log_prob, current_log_prob)

            best_lengths = current_length_is_better * curr_seq_length + ~current_length_is_better * best_lengths

            current_length_is_better = current_length_is_better.unsqueeze(1)  # shape (batch, 1)
            best_sequences = current_length_is_better * most_likely_seq + ~current_length_is_better * best_sequences

            best_backptr = current_length_is_better * debug_info["copy_backptr"] + ~current_length_is_better * best_backptr

        debug_info["copy_backptr"] = best_backptr

        if k != 1:
            # we didn't keep track of all info for the rest, so discard them. Better throw an error than
            # have it compute something wrong.
            debug_info = {"copy_backptr": debug_info["copy_backptr"]}

        return best_sequences, best_lengths, debug_info

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
                source_to_copy_mask: Optional[torch.Tensor] = None,
                alignment: Optional[torch.Tensor] = None,
                emphasis: Optional[torch.Tensor] = None,
                copyable_inputs: Optional[torch.Tensor] = None) -> Dict[
        str, torch.Tensor]:
        """

        :param metadata:
        :param source_tokens:
        :param target_tokens:
        :param source_to_copy_mask: shape (batch_size, input_seq_len, output_seq_len) source_to_copy_mask[b,i,j] = 1 if
                in batch b, the input token i is observed at output position j in the gold outputs, else 0.
                if source_to_copy_mask[b,i,j] = 1 then target_tokens[b,j] = COPY_SYMBOL
        :param alignment:
        :param emphasis: for style PTB tasks that require emphasis, this is a tensor of shape (batch_size, input_seq_len)
                that contains 1 if a word should get emphasis and 0 otherwise.
        :return:
        """
        no_grad = hasattr(self, "epoch") and self.training and random.random() < self.drop_grad_rate
        self.translation_model.requires_grad_(not no_grad)
        if self.encoder is not None:
            self.encoder.requires_grad_(not no_grad)

        # shape: (batch_size, max_input_sequence_length, encoder_input_dim)
        embedded_source = self.source_embedder(source_tokens)
        if self.source_project is not None:
            embedded_source = self.source_project(embedded_source)

        if emphasis is not None:
            embedded_source += emphasis.unsqueeze(2) * self._highlighting_param

        # shape: (batch_size, max_input_sequence_length)
        source_mask = util.get_text_field_mask(source_tokens)

        if self.encoder:
            encoder_outputs = self.encoder(embedded_source, source_mask)
            if self.rho is not None:
                if self.drop_context > 0.0 and self.training:
                    # randomly drop context but do so for entire sentences
                    encoder_outputs = (torch.rand(embedded_source.shape[0], device=embedded_source.device) < self.drop_context).unsqueeze(1).unsqueeze(2) * self.rho * encoder_outputs + embedded_source
                else:
                    encoder_outputs = self.rho*encoder_outputs + embedded_source
        else:
            encoder_outputs = embedded_source

        readable_source = [m["source_tokens"] for m in metadata]

        ret = {"source_lengths": get_lengths_from_binary_sequence_mask(source_mask),
               "readable_source": readable_source}

        if target_tokens:
            assert source_to_copy_mask is not None, "Need mask for copying"
            targets = target_tokens["tokens"]["tokens"]  # shape (batch_size, output seq length)
            target_mask = util.get_text_field_mask(target_tokens)  # shape (batch_size, output seq length)
            target_lengths = get_lengths_from_binary_sequence_mask(target_mask)
            ret["targets"] = targets
            ret["gold_target_lengths"] = get_lengths_from_binary_sequence_mask(target_mask)

            readable_target = [m["target_tokens"] for m in metadata]
            ret["readable_targets"] = readable_target

            # assert torch.all(self.oov_id != targets), "One of the target tokens was the OOV symbol. " \
            #                                           "The correctness of the copy mechanism relies on OOV to never occur."

        self.test_mode.check_callback(self)
        if not self.training:

            if self.closed_vocabulary_grammar:
                # Identify which source tokens are the same as tokens in the target vocabulary
                source2vocab = np.zeros((embedded_source.shape[0], embedded_source.shape[1], self.vocab.get_vocab_size(self.target_namespace)))
                tok2vocab = self.vocab.get_token_to_index_vocabulary(self.target_namespace)
                for b, readable in enumerate(readable_source):
                    for i, tok in enumerate(readable):
                        if tok in tok2vocab:
                            source2vocab[b, i, tok2vocab[tok]] = 1.0
                source2vocab = torch.from_numpy(source2vocab).to(embedded_source.device)
            else:
                source2vocab = None

            # Make predictions
            predictions, predicted_lengths, debug_info = self.most_likely_sequence(encoder_outputs, embedded_source,
                                                                                   source_mask, self.k,
                                                                                   use_grammar=self.dev_use_grammar,
                                                                                   copyable_inputs=copyable_inputs if self.dev_use_grammar else None,
                                                                                   source2vocab=source2vocab)
            ret["predicted_target_lengths"] = predicted_lengths
            ret["predictions"] = predictions
            ret.update(debug_info)

            readable_prediction = self.make_readable_reconstruct_backptr(predictions, predicted_lengths,
                                                                         self.target_namespace,
                                                                         ret["copy_backptr"], readable_source)

            if target_tokens:
                self.levenshtein_distance.add_instances(
                    readable_prediction,
                    readable_target)
                if self.nlg_metrics is not None:
                    self.nlg_metrics.add_instances(readable_prediction, [m["target_string"] for m in metadata])

                for m in self.metrics:
                    m.add_instances(readable_prediction, readable_target)

                self.length_acc(predicted_lengths, target_lengths)

                # print("Preds:", pred[0], gold[0], gold_mask[0])
                # self.tok_acc_wo_length(pred, gold, gold_mask)

            if self.test_mode.is_in_test_mode(self):
                # we are in test mode, run everything again but with test configuration (i.e. use grammar and use top k length search)
                predictions, predicted_lengths, debug_info = self.most_likely_sequence(encoder_outputs,
                                                                                       embedded_source,
                                                                                       source_mask, self.test_k,
                                                                                       use_grammar=self.test_use_grammar,
                                                                                       copyable_inputs=copyable_inputs,
                                                                                       source2vocab=source2vocab)
                ret["predicted_target_lengths"] = predicted_lengths
                ret["predictions"] = predictions
                ret.update(debug_info)

                readable_prediction = self.make_readable_reconstruct_backptr(predictions, predicted_lengths,
                                                                             self.target_namespace,
                                                                             ret["copy_backptr"], readable_source)
                if target_tokens:
                    self.test_mode.add_instances(readable_prediction, readable_target)

        if target_tokens and (self.training or self.compute_loss_during_dev):
            loss = torch.zeros(1, device=encoder_outputs.device)
            output_logits, d = self.compute_logits(encoder_outputs, embedded_source, source_mask, target_mask,
                                                   source_to_copy_mask)
            for k, v in d.items():
                if k != "output":
                    ret[k + "_given_length"] = v

            if "reorder_after_fertility" in d:
                self.identity_counter.add_matrix(d["reorder_after_fertility"], target_lengths)
                self.identity_counter2.add_matrix(d["reorder_after_fertility"], target_lengths)
                self.perm_entropy.add_matrix(d["reorder_after_fertility"], target_mask)
            elif "reorder_before_fertility" in d:
                source_lengths = source_mask.sum(dim=-1)  # shape (batch_size,)
                self.identity_counter.add_matrix(d["reorder_before_fertility"], source_lengths)
                self.identity_counter2.add_matrix(d["reorder_before_fertility"], source_lengths)
                self.perm_entropy.add_matrix(d["reorder_before_fertility"], source_mask)

            total_alignment = d["alignment_after_fertility_and_reorder"].sum(
                dim=-1)  # shape (batch_size, input_seq_len, output_seq_len)

            if alignment is not None:
                self.alignment_acc(total_alignment, alignment,
                                   target_mask * torch.any(alignment >= self.alignment_confidence_thresh, dim=1))

            if hasattr(self, "epoch") and self.epoch < self.pretrain_epochs and alignment is not None:
                # smoothed_alignment = (alignment + 1e-20) ** self.smooth_alignment
                # smoothed_alignment = smoothed_alignment / smoothed_alignment.sum(dim=-1, keepdims=True)

                alignment_loss = self.alignment_loss_weight * self.pretrain_loss(total_alignment, source_mask,
                                                                                 target_mask
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
                                                                        proper_log_probs=False,
                                                                        use_grammar=False)

            ret["predictions_given_length"] = top_1_predictions

            self.seq_acc(top_1_predictions.unsqueeze(1), targets, target_mask)
            self.tok_acc(output_logits, targets, target_mask)

            if self.training:
                # at test time, we find the most likely sequence and take the length of that.
                _, most_likely_lengths = self.top_k_lengths(embedded_source, source_mask, 1)  # shape (batch_size, 1)
                self.length_acc(most_likely_lengths.squeeze(1), target_lengths)

            if self.crf:
                loss += -self.crf(output_logits, targets, target_mask)
            else:
                loss += my_sequence_cross_entropy_with_logits(output_logits, targets, target_mask, label_smoothing=self.label_smoothing)

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

    def make_readable_reconstruct_backptr(self, batch_of_tokens, lengths: torch.Tensor, namespace: str,
                                          backpointrs: torch.Tensor,
                                          batch_of_source_tokens: List[List[str]]) -> List[List[str]]:
        r = []
        for batch, length_of_seq, backptr, source_sent in zip(batch_of_tokens.cpu().numpy(), lengths.cpu().numpy(),
                                                              backpointrs.cpu().numpy(),
                                                              batch_of_source_tokens):
            a = []
            for i, tok in enumerate(batch[:length_of_seq]):
                tok = self.vocab.get_token_from_index(tok, namespace=namespace)
                if tok == COPY_SYMBOL:
                    a.append(source_sent[backptr[i]])
                else:
                    a.append(tok)
            r.append(a)
        return r

    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:

        # if "targets" in output_dict:
        #     output_dict["readable_targets"] = self.make_readable(output_dict["targets"],
        #                                                          output_dict["gold_target_lengths"],
        #                                                          self.target_namespace)

        if "predictions" in output_dict:
            output_dict["readable_predictions"] = self.make_readable_reconstruct_backptr(output_dict["predictions"],
                                                                                         output_dict[
                                                                                             "predicted_target_lengths"],
                                                                                         self.target_namespace,
                                                                                         output_dict["copy_backptr"],
                                                                                         output_dict["readable_source"])

        if "predictions_given_length" in output_dict:
            output_dict["readable_predictions_given_length"] = self.make_readable(
                output_dict["predictions_given_length"], output_dict["gold_target_lengths"], self.target_namespace)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        d = {
            "seq_acc_given_length": self.seq_acc.get_metric(reset)["accuracy"],
             "tok_acc": self.tok_acc.get_metric(reset)}
        d.update(self.identity_counter.get_metrics(reset))
        d.update(self.perm_entropy.get_metrics(reset))
        d["alignment_acc"] = self.alignment_acc.get_metric(reset)
        d["length_acc"] = self.length_acc.get_metric(reset)
        d["frac_identities_thresh_0.3"] = self.identity_counter2.get_metrics(reset)["frac_identities"]
        if self.nlg_metrics is not None:
            d.update(self.nlg_metrics.get_metric(reset))
        if not self.training:
            d.update(self.levenshtein_distance.get_metric(reset))

        for m in self.metrics:
            d.update(m.get_metric(reset))

        if self.test_mode.is_in_test_mode(self):
            d.update(self.test_mode.get_metrics(reset))
        return d
