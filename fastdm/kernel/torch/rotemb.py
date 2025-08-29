import torch

from fastdm.kernel.registry import kernel_registry

@kernel_registry.register('rotembd', 'torch')
def rotary_pos_embedding_torch(query: torch.Tensor,
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
    def _apply_rotary_emb_torch(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool,
    ) -> torch.Tensor:
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)

    pos_ids = torch.arange(query.shape[1], device=query.device)
    cos, sin = cos_sin_cache.index_select(0, pos_ids).chunk(2, dim=-1)
    
    q_shape, k_shape = query.shape, key.shape

    query_rot = _apply_rotary_emb_torch(query.view(q_shape[0], q_shape[1], -1, head_size), cos, sin, is_neox)
    key_rot = _apply_rotary_emb_torch(key.view(k_shape[0], k_shape[1], -1, head_size), cos, sin, is_neox)

    query.copy_(query_rot.reshape(q_shape))
    key.copy_(key_rot.reshape(k_shape))

    return