from typing import Optional

import torch

from fastdm.kernel.registry import kernel_registry

@kernel_registry.register('sdpa', 'torch')
def sdpa_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool = False,
    scale: Optional[float] = None,
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
    Returns:
        torch.Tensor: The output tensor after applying attention.
    """
    b, t, c = query.size()

    q = query.view(query.size(0), query.size(1), num_q_heads, head_dim).transpose(1, 2)
    k = key.view(key.size(0), key.size(1), num_kv_heads, head_dim).transpose(1, 2)
    v = value.view(value.size(0), value.size(1), num_kv_heads, head_dim).transpose(1, 2)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale
        )

    attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)
    return attn_output

@kernel_registry.register('sdpa_sparse', 'torch')
def sdpa_sparse_torch(
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
    raise ValueError(f"Now sparge_attn isn't supported in kernel backend torch.")