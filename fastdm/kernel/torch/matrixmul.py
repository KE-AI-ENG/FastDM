from typing import Optional

import torch

from fastdm.kernel.registry import kernel_registry

@kernel_registry.register('fp8_matmul', 'torch')
def fp8_matmul_torch(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    '''
    Perform matrix multiplication with FP8 quantized tensors.

    Args:
        a: The first input tensor in FP8. shape:[m,k]
        b: The second input tensor in FP8. shape:[k,n]
        scale_a: Scaling factor for the first tensor. shape:[m,1]
        scale_b: Scaling factor for the second tensor. shape:[n,1]
        out_dtype: The output data type.
        bias: Optional bias tensor. shape:[n]

    Returns:
        torch.Tensor: The result of the matrix multiplication. shape: [m,n]
    
    '''
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16)
    assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype

    out = torch._scaled_mm(a, b, scale_a, scale_b.transpose(0,1), bias, out_dtype=out_dtype) #the scaled_mm need scale_a is (m,1), scale_b is (1,n)

    return out

@kernel_registry.register('int8_matmul', 'torch')
def int8_matmul_torch(a: torch.Tensor,
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
        a: The first input tensor in int8. shape:[m,k]
        b: The second input tensor in int8. shape:[k,n]
        scale_a: Scaling factor for the first tensor. shape:[m,1]
        scale_b: Scaling factor for the second tensor. shape:[n,1]
        out_dtype: The output data type.
        azp_adj: Adjusted zero-point tensor, it is weights-colsum, used for asymmetric quantization. shape:[1,n]
        azp: Zero-point tensor. shape:[m,1]
        bias: Optional bias tensor. shape:[n]

    Returns:
        torch.Tensor: The result of the matrix multiplication. shape: [m,n]
    '''
    assert (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    assert (out_dtype is torch.bfloat16 or out_dtype is torch.float16 or out_dtype is torch.float8_e4m3fn)
    #assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype
    assert bias is None or bias.dtype == out_dtype

    quant_mm_results = a.float()@b.float()
    zp_adj_mm_resualts = azp.float()@azp_adj.float()

    scale_c = scale_a.expand(scale_a.size(0), scale_b.size(0)) * scale_b.transpose(0,1).expand(scale_a.size(0), scale_b.size(0))

    out = ((quant_mm_results-zp_adj_mm_resualts)*scale_c).to(out_dtype)

    return out+bias if bias is not None else out