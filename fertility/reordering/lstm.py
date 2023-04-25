import operator
from typing import Tuple, List

import torch

def extract_lstm_minus_feature(hidden_state, i, j):
    """
    Extract span representation using lstm-minus feature,
    
    Args:
        hidden_state: Length * Batch * Hidden_size
        i, j: start and end pos, note that i, j is the 
        real index of words, discarding bos and eos:
        ... i, [i+1, ... , j+1], j+2, ...
    """
    seq_len, bs, hidden_size = hidden_state.size()
    assert hidden_size % 2 == 0
    split_point = hidden_size // 2
    hidden_f = hidden_state[j + 1, :, :split_point] - hidden_state[i, :, :split_point]
    hidden_b = (
        hidden_state[i + 1, :, split_point:] - hidden_state[j + 2, :, split_point:]
    )
    span_v = torch.cat([hidden_f, hidden_b], dim=-1)
    return span_v


def extract_all_span_features(hidden_state):
    """
    Return: ret[i] of size  n * bs * hidden_size, where n = seq_len - i
    """
    _seq_len, bs, hidden_size = hidden_state.size()
    assert hidden_size % 2 == 0  # bilstm by default
    seq_len = _seq_len - 2  # discard bos and eos

    ret = []
    for i in range(seq_len):
        temp_list = []
        for j in range(i, seq_len):
            span_v = extract_lstm_minus_feature(hidden_state, i, j)
            temp_list.append(span_v)
        temp_v = torch.stack(temp_list, dim=0)
        ret.append(temp_v)
    return ret


class SpanRepresentation(object):
    """
    Hidden state is the encoding from a LSTM; note that bos and eos 
    is attached to each sentence
    """

    def __init__(self, hidden_state, lengths):
        """
        Args:
            hidden_state: bs * length * hidden_size; we need to transform it
            into bs second for the convenience of parsing

            lengths: lengths of sequences including bos and eos, but get_lengths 
            function should return the real length
        """
        self.hidden_state = hidden_state.transpose(0, 1)  # Length * Bath * Hidden_size
        self.lengths = lengths
        self.span_v = extract_all_span_features(self.hidden_state)
        self.device = hidden_state.device

    def num_tokens(self):
        # discard bos and eos
        return self.hidden_state.size()[0] - 2

    def num_batches(self):
        return self.hidden_state.size()[1]

    def get_lengths(self):
        # convert tensor to int, remove bos and eos
        lengths = [int(l) - 2 for l in self.lengths]
        return lengths

    def get_span(self, i, j):
        """
        Obtain vector for span from i to j (exclusive)
        """
        assert j > i
        return self.span_v[i][j - i - 1]

