from typing import Optional, Tuple, Union

import torch

from fastdm.kernel.registry import kernel_registry

import triton
import triton.language as tl

# Adapted from https://github.com/sgl-project/sglang/blob/v0.5.0rc0/python/sglang/srt/layers/quantization/int8_kernel.py#L21
@triton.jit
def _per_token_quant_int8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127
    x_q = x * (127 / absmax)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


def quant_per_token_int8_sym(x, scale_dtype=torch.float32):
    assert x.ndim == 2
    assert x.is_contiguous()
    M = x.shape[0]
    N = x.shape[1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_quant_int8[(M,)](
        x,
        x_q,
        scales,
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    
    return x_q, scales



@triton.jit
def get_max_min_with_mask(
        x: tl.tensor, 
        mask,
        BLOCK: tl.constexpr
    ):
    max_inf = tl.full((BLOCK,), float("inf"), dtype=tl.float32)
    min_inf = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    
    # get v_max
    x_with_ninf = tl.where(mask, x, min_inf)
    v_max = tl.max(x_with_ninf)

    # get v_min
    x_with_inf = tl.where(mask, x, max_inf)
    v_min = tl.min(x_with_inf)

    return v_max, v_min

@triton.jit
def _per_token_quant_int8_asym(
    x_ptr,
    xq_ptr,
    scale_ptr,
    zp_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0)
    v_max, v_min = get_max_min_with_mask(x, mask, BLOCK)
    v_range = tl.maximum((v_max - v_min), 1e-10)
    scale_x = v_range / (127 + 128)
    zero = (-128 - v_min / scale_x)

    x_q = x * ((127 + 128) / v_range) + zero
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)
    zero = tl.extra.cuda.libdevice.round(zero).to(tl.int32)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))
    tl.store(zp_ptr + row_id, zero)


def quant_per_token_int8_asym(x, scale_dtype=torch.float32):
    assert x.ndim == 2
    assert x.is_contiguous()
    x = x.to(torch.float32)
    M = x.shape[0]
    N = x.shape[1]
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=scale_dtype)
    zero_points = torch.empty_like(scales).to(torch.int32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_quant_int8_asym[(M,)](
        x,
        x_q,
        scales,
        zero_points, 
        stride_x=x.stride(-2),
        stride_xq=x_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    
    return x_q, scales, zero_points


@kernel_registry.register('quantize_to_int8', 'triton')
def quantize_to_int8_triton(
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
    if symmetric:
        output, input_scales = quant_per_token_int8_sym(input)
        input_azp = None
    else:
        output, input_scales, input_azp = quant_per_token_int8_asym(input)

    return output, input_scales, input_azp



@triton.jit
def _per_token_quant_fp8(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 448.0
    x_q = x / (absmax / 448.0)
    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x.to(scale_ptr.dtype.element_ty))


@kernel_registry.register('quantize_to_fp8', 'triton')
def quantize_to_fp8_triton(
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
    assert input.ndim == 2
    assert input.is_contiguous()
    M = input.shape[0]
    N = input.shape[1]
    input_q = torch.empty_like(input, device=input.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(input.shape[:-1] + (1,), device=input.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_quant_fp8[(M,)](
        input,
        input_q,
        scales,
        stride_x=input.stride(-2),
        stride_xq=input_q.stride(-2),
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    
    return input_q, scales
