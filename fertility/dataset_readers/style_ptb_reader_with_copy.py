import csv
from typing import Dict, Optional, List
import logging
import copy

import numpy as np
from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField, TensorField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer, Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from fertility.discrete_copy_nat import COPY_SYMBOL

logger = logging.getLogger(__name__)


@DatasetReader.register("style_ptb_with_copy")
class Seq2SeqDatasetReader(DatasetReader):
    """

    """

    def __init__(
        self,
        source_tokenizer: Tokenizer = None,
        target_tokenizer: Tokenizer = None,
        source_token_indexers: Dict[str, TokenIndexer] = None,
        target_token_indexers: Dict[str, TokenIndexer] = None,
        non_copyable: Optional[List[str]] = None,
        source_add_start_token: bool = True,
        source_add_end_token: bool = True,
        target_add_start_token: bool = True,
        target_add_end_token: bool = True,
        start_symbol: str = START_SYMBOL,
        end_symbol: str = END_SYMBOL,
        delimiter: str = "\t",
        quoting: int = csv.QUOTE_MINIMAL,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs
        )
        self._source_tokenizer = source_tokenizer or SpacyTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers

        self._source_add_start_token = source_add_start_token
        self._source_add_end_token = source_add_end_token
        self._target_add_start_token = target_add_start_token
        self._target_add_end_token = target_add_end_token
        self._start_token: Optional[Token] = None
        self._end_token: Optional[Token] = None
        if (
            source_add_start_token
            or source_add_end_token
            or target_add_start_token
            or target_add_end_token
        ):
            if source_add_start_token or source_add_end_token:
                self._check_start_end_tokens(start_symbol, end_symbol, self._source_tokenizer)
            if (
                target_add_start_token or target_add_end_token
            ) and self._target_tokenizer != self._source_tokenizer:
                self._check_start_end_tokens(start_symbol, end_symbol, self._target_tokenizer)
        self._start_token = Token(start_symbol)
        self._end_token = Token(end_symbol)

        self._delimiter = delimiter
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        self.quoting = quoting
        self.non_copyable = set(non_copyable) if non_copyable is not None else set()

    @overrides
    def _read(self, file_path: str):
        # Reset exceeded counts
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line_num, row in self.shard_iterable(
                enumerate(csv.reader(data_file, delimiter=self._delimiter, quoting=self.quoting))
            ):
                if len(row) != 2:
                    raise ConfigurationError(
                        "Invalid line format: %s (line number %d)" % (row, line_num + 1)
                    )
                source_sequence, target_sequence = row
                if len(source_sequence) == 0 or len(target_sequence) == 0:
                    continue
                yield self.text_to_instance(source_sequence, target_sequence)

    @overrides
    def text_to_instance(
        self, source_string: str, target_string: str = None
    ) -> Instance:  # type: ignore
        #source_string is something like this:
        # it all adds up to a cold winter here ; cold
        # -> split on semicolon
        source_string, emphasis = source_string.split(";")
        emphasis = emphasis.strip()
        tokenized_source = self._source_tokenizer.tokenize(source_string)
        if self._source_add_start_token:
            tokenized_source.insert(0, copy.deepcopy(self._start_token))
        if self._source_add_end_token:
            tokenized_source.append(copy.deepcopy(self._end_token))
        source_field = TextField(tokenized_source)
        d = {"source_tokens": source_field, "emphasis": TensorField(np.array([w.text == emphasis for w in tokenized_source]))}

        metadata = {"source_tokens": [t.text for t in tokenized_source]}
        if target_string is not None:
            tokenized_target = self._target_tokenizer.tokenize(target_string)
            if self._target_add_start_token:
                tokenized_target.insert(0, copy.deepcopy(self._start_token))
            if self._target_add_end_token:
                tokenized_target.append(copy.deepcopy(self._end_token))

            metadata["target_tokens"] = [t.text for t in tokenized_target]
            metadata["target_string"] = target_string

            target_toks = []
            copy_mask = np.zeros((len(metadata["source_tokens"]), len(metadata["target_tokens"])))
            for i, target_tok in enumerate(metadata["target_tokens"]):
                found = False
                if target_tok not in self.non_copyable:
                    for k, source_tok in enumerate(metadata["source_tokens"]):
                        if source_tok == target_tok:
                            copy_mask[k, i] = 1.0
                            found = True
                if found:
                    target_toks.append(COPY_SYMBOL)
                else:
                    target_toks.append(target_tok)

            # print(j["input"])
            # print(target_toks)
            # print("--")
            # input()
            
            target_field = TextField([Token(x) for x in target_toks])
            d["target_tokens"] = target_field
            d["source_to_copy_mask"] = TensorField(copy_mask)

        d["metadata"] = MetadataField(metadata)
        return Instance(d)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["source_tokens"]._token_indexers = self._source_token_indexers  # type: ignore
        if "target_tokens" in instance.fields:
            instance.fields["target_tokens"]._token_indexers = self._target_token_indexers  # type: ignore

    def _check_start_end_tokens(
        self, start_symbol: str, end_symbol: str, tokenizer: Tokenizer
    ) -> None:
        """Check that `tokenizer` correctly appends `start_symbol` and `end_symbol` to the
        sequence without splitting them. Raises a `ValueError` if this is not the case.
        """

        tokens = tokenizer.tokenize(start_symbol + " " + end_symbol)
        err_msg = (
            f"Bad start or end symbol ('{start_symbol}', '{end_symbol}') "
            f"for tokenizer {self._source_tokenizer}"
        )
        try:
            start_token, end_token = tokens[0], tokens[-1]
        except IndexError:
            raise ValueError(err_msg)
        if start_token.text != start_symbol or end_token.text != end_symbol:
            raise ValueError(err_msg)
