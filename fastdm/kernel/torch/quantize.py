from typing import Optional, Tuple

import torch

from fastdm.kernel.registry import kernel_registry

@kernel_registry.register('quantize_to_int8', 'torch')
def quantize_to_int8_torch(
    input: torch.Tensor,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize(per-token sym/asym) the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert (input.ndim == 2)

    tensor_f32 = input.float()
    
    row_min = tensor_f32.min(dim=1).values
    row_max = tensor_f32.max(dim=1).values

    quant_range = [-128, 127]
    
    if symmetric:
        abs_max = torch.max(torch.abs(row_min), torch.abs(row_max))
        scales = abs_max / quant_range[-1]
        quantized = torch.clamp(torch.round(tensor_f32 / scales[:,None]), quant_range[0], quant_range[1]).to(torch.int8)
        zero_points = None
    else:
        range_val = row_max - row_min
        scales = range_val / (quant_range[1] - quant_range[0])
        zero_points = (quant_range[0] - torch.round(row_min / scales)).to(torch.int32)
        quantized = torch.clamp(torch.round(tensor_f32 / scales[:,None] + zero_points.float()[:,None]), quant_range[0], quant_range[1]).to(torch.int8)

    return quantized, scales.unsqueeze(-1), zero_points.unsqueeze(-1) if zero_points is not None else None

@kernel_registry.register('quantize_to_fp8', 'torch')
def quantize_to_fp8_torch(
    input: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize(per-channel/per-token) input tensor to FP8 and return quantized tensor and scale.

    Args:
        input: The input tensor to be quantized to FP8

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    # This code assumes batch_dim and num_tokens are flattened
    assert (input.ndim == 2)
    finfo = torch.finfo(torch.float8_e4m3fn)
    row_min = input.min(dim=1).values
    row_max = input.max(dim=1).values
    abs_max = torch.max(torch.abs(row_min), torch.abs(row_max)).clamp(min=1e-12)
    scale = abs_max.float() / finfo.max
    quantized_input = (input.float() / scale[:,None]).clamp(min=finfo.min, max=finfo.max).to(torch.float8_e4m3fn)
    return quantized_input, scale.unsqueeze(-1)