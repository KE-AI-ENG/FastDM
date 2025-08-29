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
    fp8_attn_: bool=False
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
    b, t, c = query.size()

    q = query.view(query.size(0), query.size(1), num_q_heads, head_dim).transpose(1, 2)
    k = key.view(key.size(0), key.size(1), num_kv_heads, head_dim).transpose(1, 2)
    v = value.view(value.size(0), value.size(1), num_kv_heads, head_dim).transpose(1, 2)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale
        )

    attn_output = attn_output.transpose(1, 2).contiguous().view(b, t, c)
    return attn_output