import torch

from fastdm.kernel.registry import kernel_registry

import triton
import triton.language as tl
import math

@triton.jit
def rms_norm_kernel(
    out_ptr,    # pointer to the output
    in_ptr,     # pointer to the input
    w_ptr,      # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r, # how much to increase the pointer when moving by 1 row
    x_stride_c, # how much to increase the pointer when moving by 1 col
    N,          # number of columns in X
    eps,        # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = tl.program_id(0)
    out_ptr += pid * y_stride_r
    in_ptr += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)


def rms_norm(x: torch.Tensor,
             weight: torch.Tensor,
             eps: float = 1e-6):
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert x.ndim == 2
    assert weight.ndim == 1

    M = x.shape[0]
    N = x.shape[1]

    BLOCK_SIZE = triton.next_power_of_2(N)
    y = torch.empty_like(x)

    rms_norm_kernel[M,](y, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE)

    return y


@kernel_registry.register('rmsnorm', 'triton')
def rmsnorm_triton(input: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """ Apply RMS normalization to the input tensor.

    Args:
        input: The input tensor to be normalized.
        scale: The scaling factor for normalization.
        eps: A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    (B, S, head_num, head_dim) = input.shape

    out = torch.empty_like(input)
    input_reshape = input.view(-1, head_dim)
    if scale.ndim == 2:
        scale = scale.view(-1)
    out = rms_norm(input_reshape, scale, eps).view(-1, S, head_num, head_dim)

    return out