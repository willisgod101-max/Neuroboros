import math
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Tuple

class BinActive(Function):
    """
    Binarize the activations: x -> sign(x) with Straight-Through Estimator (STE) for gradient.
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> torch.Tensor:
        return input.sign()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[None, torch.Tensor]:
        # Straight-through estimator
        return None, grad_output.clone()

class BinConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super(BinConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Binarize weights
        binary_weight = self.weight.sign()
        # Binarize activations
        binary_input = BinActive.apply(input)
        # Use binary weight and binary activation for convolution
        out = nn.functional.conv2d(binary_input, binary_weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        return out

class BinLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(BinLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features  # for Linear layer, fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Binarize weights
        binary_weight = self.weight.sign()
        # Binarize activations
        binary_input = BinActive.apply(input)
        # Linear transformation with binary weights and activations
        out = nn.functional.linear(binary_input, binary_weight)
        if self.bias is not None:
            out += self.bias
        return out

# For the BitNet-GPT, we will use a modified version of BinLinear that scales the weights and activations.
# According to BitNet paper, we use scaling factors for weights and activations.
class BitLinear(nn.Module):
    """
    BitLinear layer: binarize weights, quantize activations to int8 (or use FP32 with scaling).
    We follow the BitNet approach: weights are binarized to {-1, +1}, activations are quantized to int8.
    However, for simplicity in this implementation, we will use the binarized weights and FP32 activations
    with scaling factors as in the original BitNet paper.
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
        # Scaling factors for weights and activations (to be learned or fixed)
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
        # Binarize weights: sign(W)
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