import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Optional


class BitLinearA48(nn.Linear):
    """
    Implementation of BitNet a4.8: 4-bit Activations for 1-bit LLMs.
    Uses Hybrid Quantization: 4-bit for inputs, 8-bit + TopK Sparsification 
    for intermediate states.
    """
    def __init__(self, in_features, out_features, bias=False, sparsity_k=0.5, is_intermediate=False):
        super().__init__(in_features, out_features, bias=bias)
        self.sparsity_k = sparsity_k
        self.is_intermediate = is_intermediate
        self.eps = 1e-5

    def ste_quantize_weights(self, w):
        """Ternary weight quantization {-1, 0, 1} with STE."""
        scale = w.abs().mean() + self.eps
        w_quant = (w / scale).round().clamp(-1, 1)
        return w + (w_quant - w).detach()

    def quantize_activations_4bit(self, x):
        """INT4 activation quantization using absmean."""
        # BitNet a4.8 uses absmean for 4-bit inputs
        beta = x.abs().mean() + self.eps
        # Scale factor for 4-bit is sqrt(7) to map to [-8, 7] range
        scale = math.sqrt(7) / beta
        x_quant = (x * scale).round().clamp(-8, 7)
        return x + (x_quant / scale - x).detach()

    def quantize_and_sparsify_8bit(self, x):
        """
        INT8 Quantization + TopK Sparsification for intermediate states.
        Activates only ~55% of parameters.
        """
        # 1. 8-bit Quantization
        gamma = x.abs().max() + self.eps
        scale = 127 / gamma
        x_quant = (x * scale).round().clamp(-128, 127)
        x_ste = x + (x_quant / scale - x).detach()

        # 2. TopK Sparsification (Q-Sparse method)
        k = int(self.sparsity_k * x.shape[-1])
        topk_values, _ = torch.topk(x.abs(), k, dim=-1)
        threshold = topk_values[..., -1].unsqueeze(-1)
        mask = (x.abs() >= threshold).float()
        
        return x_ste * mask

    def forward(self, input):
        """
        Hybrid Forward Pass:
        - Input to Attention/FFN: 4-bit
        - Intermediate states (Down-proj/Output-proj): 8-bit Sparse
        """
        w_quant = self.ste_quantize_weights(self.weight)
        
        if self.is_intermediate:
            # High-outlier layers use 8-bit + TopK
            x_quant = self.quantize_and_sparsify_8bit(input)
        else:
            # Standard inputs use 4-bit
            x_quant = self.quantize_activations_4bit(input)
        
        return F.linear(x_quant, w_quant, self.bias)


def replace_with_a48(model, sparsity_k=0.5, intermediate_keywords=None):
    """Utility to swap standard BitLinear with a48 variant."""
    if intermediate_keywords is None:
        intermediate_keywords = ['down_proj', 'c_proj', 'output']
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Check if it's an intermediate layer (Down-projection or Attention Output)
            is_inter = any(kw in name for kw in intermediate_keywords)
            
            new_layer = BitLinearA48(
                module.in_features, 
                module.out_features, 
                bias=module.bias is not None,
                sparsity_k=sparsity_k,
                is_intermediate=is_inter
            )
            # Copy weights
            new_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_layer.bias.data = module.bias.data.clone()
            setattr(model, name, new_layer)
        else:
            # Only recurse if the module has children to prevent infinite recursion
            if len(list(module.children())) > 0:
                replace_with_a48(module, sparsity_k, intermediate_keywords)