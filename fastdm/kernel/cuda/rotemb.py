import torch

from fastdm.kernel.registry import kernel_registry

import fastdm.cuda_ops as cuda_ops

@kernel_registry.register('rotembd', 'cuda')
def rotary_pos_embedding_cuda(query: torch.Tensor,
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
    pos_ids = torch.arange(query.shape[1], device=query.device).repeat(query.shape[0], 1)
    cuda_ops.rotary_emb_(pos_ids, query, key, head_size, cos_sin_cache, is_neox)
    return