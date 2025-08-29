from typing import Optional

import torch

from fastdm.kernel.registry import kernel_registry

import fastdm.cuda_ops as cuda_ops

@kernel_registry.register('fp8_matmul', 'cuda')
def fp8_matmul_cuda(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16 or out_dtype is torch.float8_e4m3fn)
    assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype

    out = cuda_ops.fp8_scaled_mm_(a, b, scale_a, scale_b, out_dtype, bias)

    return out

@kernel_registry.register('int8_matmul', 'cuda')
def int8_matmul_cuda(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      azp_adj: torch.Tensor,
                      azp: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
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
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16 or out_dtype is torch.float8_e4m3fn)
    #assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype
    assert bias is None or bias.dtype == out_dtype

    out = cuda_ops.int8_scaled_mm_(a, b, scale_a, scale_b, out_dtype, azp_adj, azp, bias)

    return out