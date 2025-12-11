import torch

from fastdm.kernel.registry import kernel_registry

import triton
import triton.language as tl
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    print("Please upgrade your triton version to 3.5.1 if you want to use the triton-backend attention kernel.")

@triton.jit
def rms_norm_smallM_kernel(
    out_ptr,    # pointer to the output
    in_ptr,     # pointer to the input
    w_ptr,      # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r, # how much to increase the pointer when moving by 1 row
    x_stride_c, # how much to increase the pointer when moving by 1 col
    N: tl.constexpr,          # number of columns in X
    eps: tl.constexpr,        # epsilon to avoid division by zero
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

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + cols * x_stride_c, mask, other=0.0).to(cdtype)

    var = tl.sum(x * x, axis=0) / N + eps
    rrms = tl.rsqrt(var)  # 1 / sqrt

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (x * rrms * w).to(cdtype)
    tl.store(out_ptr + cols * y_stride_c, y, mask=mask)


@triton.jit
def rms_norm_kernel(
    desc_y,
    desc_x,
    weight_ptr,
    N: tl.constexpr,          # number of columns in X
    eps: tl.constexpr,        # epsilon to avoid division by zero
    BLOCK_SIZE_M: tl.constexpr, # number of rows processed by each program
    BLOCK_SIZE_N: tl.constexpr,
):
    if tl.constexpr(desc_x.dtype == tl.float16) or tl.constexpr(
        desc_x.dtype == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = desc_x.dtype

    pid = tl.program_id(0)
    offset_m_start = pid * BLOCK_SIZE_M
    x = desc_x.load([offset_m_start, 0]).to(cdtype)

    var = tl.sum(x * x, axis=1) / N + eps   # no shape
    rrms = tl.rsqrt(var)  # 1 / sqrt

    offsets_n = tl.arange(0, BLOCK_SIZE_N)
    w = tl.load(weight_ptr + offsets_n, mask=offsets_n<N).to(cdtype)
    y = (x * rrms[:, None] * w[None, :]).to(cdtype)
    desc_y.store([offset_m_start, 0], y)


def rms_norm(x: torch.Tensor,
             weight: torch.Tensor,
             eps: float = 1e-6):
    assert x.is_contiguous()
    assert weight.is_contiguous()
    assert x.ndim == 2
    assert weight.ndim == 1

    M = x.shape[0]
    N = x.shape[1]
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    num_M = triton.cdiv(M, BLOCK_SIZE_M)

    y = torch.empty_like(x)
    desc_y = TensorDescriptor(y, shape=y.shape, strides=y.stride(), block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    desc_x = TensorDescriptor(x, shape=x.shape, strides=x.stride(), block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    if M < BLOCK_SIZE_M:    # small M does not use TMA
        rms_norm_smallM_kernel[M,](y, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE_N)
    else:
        rms_norm_kernel[num_M,](desc_y, desc_x, weight, N, eps, BLOCK_SIZE_M, BLOCK_SIZE_N)

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
    head_dim = input.shape[-1]

    out = torch.empty_like(input)
    input_reshape = input.view(-1, head_dim)
    if scale.ndim == 2:
        scale = scale.view(-1)
    out = rms_norm(input_reshape, scale, eps).view(input.shape)

    return out