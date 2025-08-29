import torch
import triton
import triton.language as tl
from fastdm.kernel.registry import kernel_registry


@triton.jit
def rotary_pos_emb_qk_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    out_k_ptr,
    seq_len,    # qk seq_len
    head_size,  # qk head_size
    dim_size,   # qk dim_size
    stride_qk_s,
    stride_qk_h,
    stride_qk_d,
    stride_cs_s,
    stride_cs_d,
    QK_BLOCK_SIZE: tl.constexpr,   # next_pow_of_2(dim)
    CS_BLOCK_SIZE: tl.constexpr,   # next_pow_of_2(dim//2)
):
    blockid_s = tl.program_id(axis=0)
    blockid_h = tl.program_id(axis=1)

    qk_start_offs = blockid_s*stride_qk_s + blockid_h*stride_qk_h
    qk_2i_offset = qk_start_offs + tl.arange(0, CS_BLOCK_SIZE) * 2
    qk_2i1_offset = qk_start_offs + tl.arange(0, CS_BLOCK_SIZE) * 2 + 1
    mask_qk_2i = (qk_2i_offset >= qk_start_offs) & (qk_2i_offset < qk_start_offs+dim_size)
    mask_qk_2i1 = (qk_2i1_offset >= qk_start_offs) & (qk_2i1_offset < qk_start_offs+dim_size)

    cs_start_offs = blockid_s*stride_cs_s
    cs_offset = cs_start_offs + tl.arange(0, CS_BLOCK_SIZE)
    mask_cs = (cs_offset >= cs_start_offs) & (cs_offset < cs_start_offs+dim_size//2)

    q_2i = tl.load(q_ptr+qk_2i_offset, mask_qk_2i)
    q_2i1 = tl.load(q_ptr+qk_2i1_offset, mask_qk_2i1)
    k_2i = tl.load(k_ptr+qk_2i_offset, mask_qk_2i)
    k_2i1 = tl.load(k_ptr+qk_2i1_offset, mask_qk_2i1)
    cos = tl.load(cos_ptr+cs_offset, mask_cs)
    sin = tl.load(sin_ptr+cs_offset, mask_cs)
    
    out_q_2i = q_2i * cos - q_2i1 * sin
    out_q_2i1 = q_2i1 * cos + q_2i * sin
    out_k_2i = k_2i * cos - k_2i1 * sin
    out_k_2i1 = k_2i1 * cos + k_2i * sin
    out_q = tl.interleave(out_q_2i, out_q_2i1)
    out_k = tl.interleave(out_k_2i, out_k_2i1)

    out_qk_offset = qk_start_offs + tl.arange(0, QK_BLOCK_SIZE)
    mask_out_qk = (out_qk_offset >= out_qk_offset) & (out_qk_offset < out_qk_offset+dim_size)
    tl.store(out_q_ptr+out_qk_offset, out_q, mask_out_qk)
    tl.store(out_k_ptr+out_qk_offset, out_k, mask_out_qk)

def rotary_pos_embedding(
        q: torch.Tensor,    # shape[seq_len, head_size, dim_size]
        k: torch.Tensor,    # shape[seq_len, head_size, dim_size]
        cos: torch.Tensor,  # shape[seq_len, dim_size]
        sin: torch.Tensor   # shape[seq_len, dim_size]
    ) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.is_contiguous() == True
    assert k.is_contiguous() == True
    assert q.ndim == 3
    assert k.ndim == 3
    assert q.shape[-1] % 2 == 0
    assert k.shape[-1] == q.shape[-1]
    assert cos.is_contiguous() == True
    assert sin.is_contiguous() == True
    assert cos.shape[0] == q.shape[0]
    assert sin.shape[0] == q.shape[0]
    assert cos.shape[-1] == q.shape[-1] // 2
    assert sin.shape[-1] == q.shape[-1] // 2

    out_q = torch.empty_like(q)
    out_k = torch.empty_like(k)

    seq_len = q.shape[0]
    head_size = q.shape[1]
    dim_size = q.shape[2]
    QK_BLOCK_SIZE = triton.next_power_of_2(dim_size)
    CS_BLOCK_SIZE = triton.next_power_of_2(dim_size//2)

    grid = (
        seq_len,
        head_size,
    )
    rotary_pos_emb_qk_kernel[grid](
        q,
        k,
        cos,
        sin,
        out_q,
        out_k,
        seq_len=seq_len,
        head_size=head_size,
        dim_size=dim_size,
        stride_qk_s=q.stride(0),
        stride_qk_h=q.stride(1),
        stride_qk_d=q.stride(2),
        stride_cs_s=cos.stride(0),
        stride_cs_d=cos.stride(1),
        QK_BLOCK_SIZE=QK_BLOCK_SIZE,
        CS_BLOCK_SIZE=CS_BLOCK_SIZE,
    )

    return out_q, out_k


@kernel_registry.register('rotembd', 'triton')
def rotary_pos_embedding_triton(query: torch.Tensor,
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
    pos_ids = torch.arange(query.shape[1], device=query.device)
    cos, sin = cos_sin_cache.index_select(0, pos_ids).chunk(2, dim=-1)
    cos = cos.contiguous()
    sin = sin.contiguous()
    query_rot, key_rot = rotary_pos_embedding(
        query.view(query.shape[0], query.shape[1], -1, head_size).squeeze(0), 
        key.view(key.shape[0], key.shape[1], -1, head_size).squeeze(0), 
        cos, sin)

    query.copy_(query_rot.reshape(query.shape))
    key.copy_(key_rot.reshape(key.shape))
    return