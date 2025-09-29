# Use the distribution mechanism to define a unified interface.
# These functions body will not be directly called; actual calls will be dispatched to the specific implementation.

from typing import Optional, Tuple

import torch
from fastdm.kernel.registry import kernel_registry

@kernel_registry.dispatch('rmsnorm')
def rms_norm(input: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """ Apply RMS normalization to the input tensor.

    Args:
        input: The input tensor to be normalized.
        scale: The scaling factor for normalization.
        eps: A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    return NotImplemented

@kernel_registry.dispatch('rotembd')
def rotary_pos_embedding(query: torch.Tensor,
                        key: torch.Tensor,
                        head_size: int,
                        cos_sin_cache: torch.Tensor,
                        is_neox: bool = False):
    '''
    Apply rotary embedding to keys and queries with precomputed cos/sin values.
    This is designed to be compatible with the SGL/vLLM implementation.

    Parameters
    ----------
    query : torch.Tensor
        Query tensor, shape: ``(batch, seq_len, num_q_heads * head_size)``.
    key : torch.Tensor
        Key tensor, shape: ``(batch, seq_len, num_k_heads * head_size)``.
    cos_sin_cache : torch.Tensor
        Cosine and Sine cache tensor, shape: ``(max_seq_len, rotary_dim)``.
        Cosine is the first half and Sine is the second half on rotary_dim.
    is_neox : bool
        Whether to use Neox style RoPE, default: ``True``.

        * If ``True``, the last dimension of the query/key tensor is not interleaved, i.e.,
          we rorate the first half dimensions ``([..., :head_dim//2])`` and the second half
          dimensions ``([..., head_dim//2:])``.

        * If ``False``, the last dimension of the query/key tensor is interleaved, i.e.,
          we rotate the even dimensions ``([..., ::2])`` and odd dimensions ``([..., 1::2])``.
    '''
    return NotImplemented

@kernel_registry.dispatch('gelu_and_mul', force_backend="triton") # Force to use triton backend for this operation, the cuda backend is not implemented yet.
def gelu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """
    The gelu-and-mul here is somewhat different from the large language model. Its scope is opposite.
    It apply GELU activation to the second half of the input tensor and multiply it with the first half.
    x = x[:d] * GELU(x[d:]) where d = x.shape[-1] // 2.

    Args:
        input: The input tensor to be processed.

    Returns:
        torch.Tensor: The output tensor after applying GELU and multiplication.
    """
    return NotImplemented

@kernel_registry.dispatch('quantize_to_int8')
def quantize_to_int8(
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
    return NotImplemented

@kernel_registry.dispatch('quantize_to_fp8')
def quantize_to_fp8(
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
    return NotImplemented

@kernel_registry.dispatch('fp8_matmul')
def fp8_matmul(a: torch.Tensor,
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
    return NotImplemented

@kernel_registry.dispatch('int8_matmul')
def int8_matmul(a: torch.Tensor,
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

    return NotImplemented

@kernel_registry.dispatch('sdpa')
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool = False,
    scale: Optional[float] = None,
    fp8_attn_: bool=False,
    sparge_attn_: bool = False, 
    sparse_mask: Optional[torch.Tensor] = None, 
) -> torch.Tensor:
    """
    Perform scaled dot-product attention.
    Args:
        query: Query tensor of shape (batch_size, seq_len, num_q_heads * head_dim).
        key: Key tensor of shape (batch_size, seq_len, num_kv_heads * head_dim).
        value: Value tensor of shape (batch_size, seq_len, num_kv_heads * head_dim).
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension of each head.
        is_causal: Whether to apply causal masking.
        scale: Scaling factor for the attention scores.
        fp8_attn_: for hopper gpu, we use fp8-attn to get better performance. if your generation results is worse than baseline, please disable it.
    Returns:
        torch.Tensor: The output tensor after applying attention.
    """
    return NotImplemented

@kernel_registry.dispatch('sdpa_sparse')
def sparse_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool = False,
    scale: Optional[float] = None,
    sparse_mask: Optional[torch.Tensor] = None, 
) -> torch.Tensor:
    """
    Perform scaled dot-product attention.
    Args:
        query: Query tensor of shape (batch_size, seq_len, num_q_heads * head_dim).
        key: Key tensor of shape (batch_size, seq_len, num_kv_heads * head_dim).
        value: Value tensor of shape (batch_size, seq_len, num_kv_heads * head_dim).
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension of each head.
        is_causal: Whether to apply causal masking.
        scale: Scaling factor for the attention scores.
        sparse_mask: Mask_id for sparge_attn, shape (batch_size, num_q_heads, seq_len//BLOCK_Q, seq_len//BLOCK_K). 1 for compute, 0 for skip.
    Returns:
        torch.Tensor: The output tensor after applying attention.
    """
    return NotImplemented

# Add more ops unified interfaces...