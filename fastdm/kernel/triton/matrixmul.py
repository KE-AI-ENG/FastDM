from typing import Optional

import torch
import triton
import triton.language as tl
from fastdm.kernel.registry import kernel_registry

@triton.jit
def matmul_fp8_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    s_a_ptr, 
    s_b_ptr, 
    with_bias: tl.constexpr,
    bias_ptr, 
    M,
    N,
    K,
    a_stride_m, 
    a_stride_k,
    b_stride_k, 
    b_stride_n,
    c_stride_m, 
    c_stride_n, 
    s_a_stride_m,
    s_b_stride_n,
    bias_stride_n,
    BM: tl.constexpr = 128, 
    BN: tl.constexpr = 128, 
    BK: tl.constexpr = 128
):
    blockid_m = tl.program_id(axis=0)
    blockid_n = tl.program_id(axis=1)
    offset_start_m = blockid_m * BM
    offset_m = offset_start_m + tl.arange(0, BM)
    offset_start_n = blockid_n * BN
    offset_n = offset_start_n + tl.arange(0, BN)  

    c = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BK)):
        offset_start_k = kk * BK
        offset_k = offset_start_k + tl.arange(0, BK)
        mask_a = (offset_m[:, None] < M) & (offset_k[None, :] < K)
        a_offset = offset_m[:, None] * a_stride_m + offset_k[None,:] * a_stride_k
        a = tl.load(a_ptr+a_offset, mask_a, other=0.0)

        mask_b = (offset_k[:, None] < K) & (offset_n[None, :] < N)
        b_offset = offset_k[:, None] * b_stride_k + offset_n[None, :] * b_stride_n
        b = tl.load(b_ptr+b_offset, mask_b, other=0.0)

        c += tl.dot(a, b)

    s_a_offset = offset_start_m + tl.arange(0, BM) * s_a_stride_m
    mask_s_a = s_a_offset < M
    s_a = tl.load(s_a_ptr+s_a_offset, mask_s_a, other=0.0)
    s_b_offset = offset_start_n + tl.arange(0, BN) * s_b_stride_n
    mask_s_b_bias = s_b_offset < N
    s_b = tl.load(s_b_ptr+s_b_offset, mask_s_b_bias, other=0.0)
    c = c * s_a[:, None] * s_b
    if with_bias == True:
        bias_offset = offset_start_n + tl.arange(0, BN) * bias_stride_n
        bias = tl.load(bias_ptr+bias_offset, mask_s_b_bias, other=0.0)
        c += bias

    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    c_offset = offset_m[:, None] * c_stride_m + offset_n[None, :] * c_stride_n
    tl.store(c_ptr + c_offset, c, mask_c)


@kernel_registry.register('fp8_matmul', 'triton')
def fp8_matmul_triton(a: torch.Tensor,                      # [M, K]
                      b: torch.Tensor,                      # [K, N]
                      scale_a: torch.Tensor,                # [M, 1]
                      scale_b: torch.Tensor,                # [N, 1]
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None   # [N]
    ) -> torch.Tensor:
    '''
    Perform matrix multiplication with FP8 quantized tensors.

    Args:
        a: The first input tensor in FP8.
        b: The second input tensor in FP8.
        scale_a: Scaling factor for the first tensor.
        scale_b: Scaling factor for the second tensor.
        out_dtype: The output data type.
        bias: Optional bias tensor.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    
    '''
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert scale_a.ndim == 2
    assert scale_b.ndim == 2
    assert scale_a.shape[0] == M and scale_a.shape[1] == 1
    assert scale_b.shape[0] == N and scale_b.shape[1] == 1
    if bias != None:
        assert bias.ndim == 1
        assert bias.shape[0] == N
        with_bias = True
    else:
        with_bias = False

    c = torch.empty(M, N, device="cuda", dtype=out_dtype)

    grid = lambda mata: (triton.cdiv(M, mata["BM"]), triton.cdiv(N,  mata["BN"]))

    matmul_fp8_kernel[grid](
                    a, b, c, 
                    scale_a, scale_b, 
                    with_bias, bias,
                    M, N, K, 
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), scale_a.stride(0), scale_b.stride(0), bias.stride(0) if bias!=None else -1
                    )

    return c


@triton.jit
def matmul_int8_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr, 
    s_a_ptr, 
    s_b_ptr, 
    azp_ptr,
    azp_adj_ptr, 
    with_bias: tl.constexpr,
    bias_ptr, 
    M,
    N,
    K,
    a_stride_m, 
    a_stride_k,
    b_stride_k, 
    b_stride_n,
    c_stride_m, 
    c_stride_n, 
    s_a_stride_m,
    s_b_stride_n,
    azp_stride_m,
    azp_adj_stride_n,
    bias_stride_n,
    BM: tl.constexpr = 128, 
    BN: tl.constexpr = 128, 
    BK: tl.constexpr = 128
):
    blockid_m = tl.program_id(axis=0)
    blockid_n = tl.program_id(axis=1)
    offset_start_m = blockid_m * BM
    offset_m = offset_start_m + tl.arange(0, BM)
    offset_start_n = blockid_n * BN
    offset_n = offset_start_n + tl.arange(0, BN)  

    c = tl.zeros((BM, BN), dtype=tl.int32)
    for kk in range(0, tl.cdiv(K, BK)):
        offset_start_k = kk * BK
        offset_k = offset_start_k + tl.arange(0, BK)
        mask_a = (offset_m[:, None] < M) & (offset_k[None, :] < K)
        a_offset = offset_m[:, None] * a_stride_m + offset_k[None,:] * a_stride_k
        a = tl.load(a_ptr+a_offset, mask_a, other=0)

        mask_b = (offset_k[:, None] < K) & (offset_n[None, :] < N)
        b_offset = offset_k[:, None] * b_stride_k + offset_n[None, :] * b_stride_n
        b = tl.load(b_ptr+b_offset, mask_b, other=0)

        c += tl.dot(a, b)

    s_a_azp_offset = offset_start_m + tl.arange(0, BM) * s_a_stride_m
    mask_s_a_azp = s_a_azp_offset < M
    s_a = tl.load(s_a_ptr+s_a_azp_offset, mask_s_a_azp, other=0.0)
    s_b_azp_adj_bias_offset = offset_start_n + tl.arange(0, BN) * s_b_stride_n
    mask_s_b_azp_adj_bias = s_b_azp_adj_bias_offset < N
    s_b = tl.load(s_b_ptr+s_b_azp_adj_bias_offset, mask_s_b_azp_adj_bias, other=0.0)
    azp = tl.load(azp_ptr+s_a_azp_offset, mask_s_a_azp, other=0)
    azp_adj = tl.load(azp_adj_ptr+s_b_azp_adj_bias_offset, mask_s_b_azp_adj_bias, other=0)
    c -= azp[:, None] * azp_adj[None, :]
    c = c * s_a[:, None] * s_b
    
    if with_bias == True:
        bias = tl.load(bias_ptr+s_b_azp_adj_bias_offset, mask_s_b_azp_adj_bias, other=0.0)
        c = c.to(tl.bfloat16) + bias

    mask_c = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    c_offset = offset_m[:, None] * c_stride_m + offset_n[None, :] * c_stride_n
    tl.store(c_ptr + c_offset, c, mask_c)


@kernel_registry.register('int8_matmul', 'triton')
def int8_matmul_triton(a: torch.Tensor,                     # [M, K]
                      b: torch.Tensor,                      # [K, N]
                      scale_a: torch.Tensor,                # [M, 1]
                      scale_b: torch.Tensor,                # [N, 1]
                      out_dtype: torch.dtype,
                      azp_adj: torch.Tensor,                # [1, N]
                      azp: torch.Tensor,                    # [M, 1]
                      bias: Optional[torch.Tensor] = None   # [N]
    ) -> torch.Tensor:
    '''
    Perform matrix multiplication with int8 quantized tensors.

    Args:
        a: The first input tensor in int8.
        b: The second input tensor in int8.
        scale_a: Scaling factor for the first tensor.
        scale_b: Scaling factor for the second tensor.
        out_dtype: The output data type.
        azp_adj: Adjusted zero-point tensor, it is weights-colsum, used for asymmetric quantization.
        azp: Zero-point tensor.
        bias: Optional bias tensor.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    '''
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.dtype == torch.int8
    assert b.dtype == torch.int8
    assert scale_a.ndim == 2
    assert scale_b.ndim == 2
    assert scale_a.shape[0] == M and scale_a.shape[1] == 1
    assert scale_b.shape[0] == N and scale_b.shape[1] == 1
    assert azp_adj.ndim == 2
    assert azp.ndim == 2
    assert azp_adj.shape[0] == 1 and azp_adj.shape[1] == N
    assert azp.shape[0] == M and azp.shape[1] == 1
    assert scale_a.stride(0) == azp.stride(0)
    assert scale_b.stride(0) == azp_adj.stride(1)
    if bias != None:
        assert bias.ndim == 1
        assert bias.shape[0] == N
        assert scale_b.stride(0) == bias.stride(0)
        with_bias = True
    else:
        with_bias = False

    c = torch.empty(M, N, device="cuda", dtype=out_dtype)

    grid = lambda mata: (triton.cdiv(M, mata["BM"]), triton.cdiv(N,  mata["BN"]))

    matmul_int8_kernel[grid](
                    a, b, c, 
                    scale_a, scale_b, 
                    azp, azp_adj, 
                    with_bias, bias,
                    M, N, K, 
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), 
                    scale_a.stride(0), scale_b.stride(0), azp.stride(0), azp_adj.stride(1), 
                    bias.stride(0) if bias!=None else -1
                    )

    return c