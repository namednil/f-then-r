import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from allennlp.nn import Activation
from allennlp.nn.util import get_device_of
from torch.nn import Module

def scale_grad(t: torch.Tensor, s: float):
    detached = t.detach()
    zero_with_grad = t - detached
    return s*zero_with_grad + detached

def sinusoidal_pos_embedding(d_model: int, max_len: int = 5000, pos_offset: int = 0,
                             device: Optional[torch.device] = None):
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1) + pos_offset
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def concat_sum(seq1 : torch.Tensor, seq2 : torch.Tensor, mask1 : Optional[torch.Tensor] = None, mask2 : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    @type seq1: first sequence, shape (batch_size, seq_len1, input_dim1)
    @type seq2: second sequence, shape (batch_size, seq_len2, input_dim2)
    @type mask1: mask of first sequence, shape (batch_size, seq_len1)
    @type mask2: mask of second sequence, shape (batch_size, seq_len2)

    @return a tuple with the concatenation of every pair passed through the FF + corresponding mask.
        Shapes: (batch_size, seq_len1, seq_len2, hidden_size)
                (batch_size, seq_len1, seq_len2)
                An element is masked if at least one of the corresponding vectors was masked.
    """

    batch_size, seq_len1, hidden_dim = seq1.shape
    _, seq_len2, _ = seq2.shape

    concatenated = seq1.repeat_interleave(seq_len2,0).reshape(batch_size, seq_len2, seq_len1, hidden_dim).transpose(1,2) \
                   + seq2.repeat_interleave(seq_len1,0).reshape(batch_size, seq_len1, seq_len2, hidden_dim)

    mask = None
    if mask1 is not None and mask2 is not None:
        m1_repeated = mask1.repeat_interleave(seq_len2,0).reshape(batch_size, seq_len2, seq_len1).transpose(1,2)
        m2_repeated = mask2.repeat_interleave(seq_len1,0).reshape(batch_size, seq_len1, seq_len2)
        mask =  m1_repeated & m2_repeated

    return concatenated, mask

class ConcatMLP(Module):
    """
    Intuitively, takes two list of (batched) vectors and creates an output tensor
    that contains the result of concatenating every pair and feeding it through a feed-forward neural network.
    If you set the activation to be linear and bias=False, you simply get a trainable matrix multiplication instead of fully-fledged feed-forward pass.
    """

    def __init__(self, hidden_size : int, input_dim1 : int, input_dim2 : int, activation : Activation, bias : bool = True):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_size = hidden_size
        self.activation = activation

        self.W1 = nn.Linear(input_dim1, hidden_size, bias = bias)
        self.W2 = nn.Linear(input_dim2, hidden_size, bias = False)

    def forward(self, seq1 : torch.Tensor, seq2 : torch.Tensor, mask1 : Optional[torch.Tensor] = None, mask2 : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        @type seq1: first sequence, shape (batch_size, seq_len1, input_dim1)
        @type seq2: second sequence, shape (batch_size, seq_len2, input_dim2)
        @type mask1: mask of first sequence, shape (batch_size, seq_len1)
        @type mask2: mask of second sequence, shape (batch_size, seq_len2)

        @return a tuple with the concatenation of every pair passed through the FF + corresponding mask.
            Shapes: (batch_size, seq_len1, seq_len2, hidden_size)
                    (batch_size, seq_len1, seq_len2)
                    An element is masked if at least one of the corresponding vectors was masked.
        """

        concatenated, mask = concat_sum(self.W1(seq1), self.W2(seq2), mask1, mask2)
        return self.activation(concatenated), mask


class RelativeConcatMLP(Module):
    """
    Intuitively, takes two list of (batched) vectors and creates an output tensor
    that contains the result of concatenating every pair and feeding it through a feed-forward neural network.
    If you set the activation to be linear and bias=False, you simply get a trainable matrix multiplication instead of fully-fledged feed-forward pass.
    """

    def __init__(self, hidden_size : int, input_dim1 : int, input_dim2 : int, activation : Activation, bias : bool = True, max_len: int = 300):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_size = hidden_size
        self.activation = activation

        self.W1 = nn.Linear(input_dim1, hidden_size, bias = bias)
        self.W2 = nn.Linear(input_dim2, hidden_size, bias = False)
        self.W3 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.05)

        assert hidden_size % 2 == 0

        relative_distance_encoding = torch.zeros((1, max_len, max_len, hidden_size))
        pos_end = sinusoidal_pos_embedding(hidden_size, max_len)
        for i in range(max_len):
            for j in range(max_len):
                relative_distance_encoding[0, i, j, :] = pos_end[abs(j-i)] #+ (pos_end[0] if i < j else 0) #TODO: remove the last bit.

        self.register_buffer("relative_distance_encoding", relative_distance_encoding, persistent=False)

    def forward(self, seq1 : torch.Tensor, seq2 : torch.Tensor, mask1 : Optional[torch.Tensor] = None, mask2 : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        @type seq1: first sequence, shape (batch_size, seq_len1, input_dim1)
        @type seq2: second sequence, shape (batch_size, seq_len2, input_dim2)
        @type mask1: mask of first sequence, shape (batch_size, seq_len1)
        @type mask2: mask of second sequence, shape (batch_size, seq_len2)

        @return a tuple with the concatenation of every pair passed through the FF + corresponding mask.
            Shapes: (batch_size, seq_len1, seq_len2, hidden_size)
                    (batch_size, seq_len1, seq_len2)
                    An element is masked if at least one of the corresponding vectors was masked.
        """

        concatenated, mask = concat_sum(self.W1(seq1), self.W2(seq2), mask1, mask2)
        l1 = seq1.shape[1]
        l2 = seq2.shape[1]
        concatenated = self.activation(concatenated) + self.relative_distance_encoding[:, :l1, :l2]

        return self.activation(self.W3(concatenated)), mask