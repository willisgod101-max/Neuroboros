import math
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Tuple


class BitLinear(nn.Module):
    """
    BitLinear layer: binarize weights to ternary (-1, 0, 1), scale activations.
    We follow the BitNet approach: weights are binarized to {-1, 0, 1} via sign(),
    activations are scaled by a learned scaling factor (no quantization for simplicity).
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # Scaling factors for weights and activations (to be learned)
        self.weight_scale = nn.Parameter(torch.Tensor(out_features, 1))
        self.input_scale = nn.Parameter(torch.Tensor(1, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Initialize scaling factors to 1.0
        nn.init.ones_(self.weight_scale)
        nn.init.ones_(self.input_scale)
        if self.bias is not None:
            fan_in = self.in_features  # for Linear layer, fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Binarize weights: sign(W) -> ternary {-1, 0, 1}
        binary_weight = self.weight.sign()
        # Scale the binary weights
        scaled_weight = binary_weight * self.weight_scale
        # Scale the input activations
        scaled_input = input * self.input_scale
        # Linear transformation
        out = nn.functional.linear(scaled_input, scaled_weight)
        if self.bias is not None:
            out += self.bias
        return out