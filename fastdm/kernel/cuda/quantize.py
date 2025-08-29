from typing import Optional, Tuple, Union

import torch

from fastdm.kernel.registry import kernel_registry

import fastdm.cuda_ops as cuda_ops

@kernel_registry.register('quantize_to_int8', 'cuda')
def quantize_to_int8_cuda(
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
    output = torch.empty_like(input, dtype=torch.int8)

    input_scales = torch.empty((input.numel() // input.shape[-1], 1), device=input.device, dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales, dtype=torch.int32)
    cuda_ops.int8_quant_(output, input, input_scales, input_azp)
    return output, input_scales, input_azp

@kernel_registry.register('quantize_to_fp8', 'cuda')
def quantize_to_fp8_cuda(
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
    shape: Union[Tuple[int, int], torch.Size] = input.shape

    out_dtype: torch.dtype = torch.float8_e4m3fn
    output = torch.empty(shape, device=input.device, dtype=out_dtype)

    scale = torch.empty((shape[0], 1), device=input.device, dtype=torch.float32)
    cuda_ops.fp8_quant_(output, input, scale, None)

    return output, scale