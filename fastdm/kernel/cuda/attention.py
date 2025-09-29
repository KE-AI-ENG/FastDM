from typing import Optional

import torch
import torch.nn.functional as F

from torch.nn.attention import sdpa_kernel, SDPBackend

pre_ampere = torch.cuda.get_device_capability()[0] < 8

try:
    import flashinfer
    use_flashinfer = True if pre_ampere else False
except ImportError:
    use_flashinfer = False
sdpa_backend = SDPBackend.CUDNN_ATTENTION if torch.cuda.get_device_capability()[0] >= 8 else SDPBackend.EFFICIENT_ATTENTION

#According my tests, the cudnn-attention backend can get the better performance on Hopper and Ampere GPUs than flash-attention, it requires PyTorch12.7/CUDA 12.4 or later.

try:
    from sageattention import sageattn
    use_sageattn = True if torch.cuda.get_device_capability()[0] >= 8 else False #sage-attention only supports Post-Ampere GPUs
except ImportError:
    use_sageattn = False

try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda
    use_spargeattn = True if torch.cuda.get_device_capability()[0] >= 8 else False #sparge-attention only supports Post-Ampere GPUs
except ImportError:
    use_spargeattn = False

from fastdm.kernel.registry import kernel_registry

import fastdm.cuda_ops as cuda_ops

def sageattn_wrapper(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        scale: float
    ) -> torch.Tensor:
    """ dispatch larger seq_len to avoid core dumped """
    b, t, num_q_heads, head_dim = q.shape

    total_token_num = b*t*num_q_heads
    token_num_per_kernel = 1*489830*34*(128//head_dim)
    if (head_dim in [64,128]) and (total_token_num > token_num_per_kernel): # if seq_len is extremely large
        assert num_q_heads == k.shape[2]
        assert num_q_heads == v.shape[2]

        head_num_per_kernel = token_num_per_kernel//b//t
        assert head_num_per_kernel > 0
        # split head_num
        split_section = []
        tmp = num_q_heads
        while(tmp > 0):
            split_section.append(head_num_per_kernel if tmp-head_num_per_kernel>=0 else tmp)
            tmp -= head_num_per_kernel
        q_set = torch.split(q, split_section, dim=2)
        k_set = torch.split(k, split_section, dim=2)
        v_set = torch.split(v, split_section, dim=2)

        o_set = []
        for i in range(len(q_set)):
            q_part = q_set[i].contiguous()
            k_part = k_set[i].contiguous()
            v_part = v_set[i].contiguous()
            o_part = sageattn(q_part, k_part, v_part, tensor_layout="NHD", is_causal=False, sm_scale=scale)
            o_set.append(o_part)

        attn_output = torch.cat(o_set, dim=2)

    else: 
        attn_output = sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=scale)

    return attn_output

def spargeattn_wrapper(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        scale: float,
        sparse_mask: torch.Tensor = None, 
    ) -> torch.Tensor:
    """ dispatch larger seq_len to avoid core dumped, and do padding for seq_len """
    b, t, h, d = q.shape
    _, s, _, _ = k.shape

    # padding
    if torch.cuda.get_device_capability() == (9,0):
        BLOCK_Q = 64
        BLOCK_K = 128
    else:
        BLOCK_Q = 128
        BLOCK_K = 64
    if s < BLOCK_K*4:
        assert s % BLOCK_K == 0, f"When s < {BLOCK_K*4}, the sequence length to be multiples of {BLOCK_K} for key/value!"   # for Hopper
    if (t % BLOCK_Q != 0) or (s % BLOCK_K != 0):
        t_new = ((t + BLOCK_Q - 1) // BLOCK_Q) * BLOCK_Q
        s_new = ((s + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
        q = F.pad(q, (0,0,0,0,0,t_new - t), mode='constant', value=0)
        k = F.pad(k, (0,0,0,0,0,s_new - s), mode='constant', value=0)
        v = F.pad(v, (0,0,0,0,0,s_new - s), mode='constant', value=0)
        sparse_mask = F.pad(sparse_mask, (0, s_new//BLOCK_K - s//BLOCK_K, 0, t_new//BLOCK_Q - t//BLOCK_Q), mode='constant', value=1)    # need compute

    # dispatch larger seq_len to avoid core dumped
    total_token_num = b*t*h
    token_num_per_kernel = 1*489830*34*(128//d)
    if (d in [64,128]) and (total_token_num > token_num_per_kernel): # if seq_len is extremely large
        assert h == k.shape[2]
        assert h == v.shape[2]

        head_num_per_kernel = token_num_per_kernel//b//t
        assert head_num_per_kernel > 0
        # split head_num
        split_section = []
        tmp = h
        while(tmp > 0):
            split_section.append(head_num_per_kernel if tmp-head_num_per_kernel>=0 else tmp)
            tmp -= head_num_per_kernel
        q_set = torch.split(q, split_section, dim=2)
        k_set = torch.split(k, split_section, dim=2)
        v_set = torch.split(v, split_section, dim=2)
        mask_set = torch.split(sparse_mask, split_section, dim=1)

        o_set = []
        for i in range(len(q_set)):
            q_part = q_set[i].contiguous()
            k_part = k_set[i].contiguous()
            v_part = v_set[i].contiguous()
            mask_part = mask_set[i].contiguous()
            # o_part = sageattn(q_part, k_part, v_part, tensor_layout="NHD", is_causal=False, sm_scale=scale)
            o_part = block_sparse_sage2_attn_cuda(
                    q_part, k_part, v_part, 
                    mask_id=mask_part, 
                    scale=scale, pvthreshd=20, attention_sink=False, tensor_layout="NHD", return_sparsity=False)
            o_set.append(o_part)

        attn_output = torch.cat(o_set, dim=2)

    else: 
        # attn_output = sageattn(q, k, v, tensor_layout="NHD", is_causal=False, sm_scale=scale)
        attn_output = block_sparse_sage2_attn_cuda(
                        q, k, v, 
                        mask_id=sparse_mask, 
                        scale=scale, pvthreshd=20, attention_sink=False, tensor_layout="NHD", return_sparsity=False)
    
    return attn_output[:, :t, :, :] # for padding
                
@kernel_registry.register('sdpa', 'cuda')
def sdpa_cuda(
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

    if use_flashinfer:
        q_nhd = query.view(b, t, num_q_heads, head_dim).transpose(0, 1).contiguous().view(t, -1, head_dim)
        # (batch_size * num_heads, seq_len, head_dim)
        k_hnd = key.view(b, key.size(1), num_kv_heads, head_dim).transpose(1, 2).contiguous().view(-1, key.size(1), head_dim)
        # (batch_size * num_heads, seq_len, head_dim)
        v_hnd = value.view(b, value.size(1), num_kv_heads, head_dim).transpose(1, 2).contiguous().view(-1, value.size(1), head_dim)

        if pre_ampere and query.dtype==torch.bfloat16: #the Turing GPU does not support flash-attention with bfloat16
            q_nhd = q_nhd.to(torch.float16)
            k_hnd = k_hnd.to(torch.float16)
            v_hnd = v_hnd.to(torch.float16)

        attn_output = flashinfer.single_prefill_with_kv_cache(q_nhd, k_hnd, v_hnd, kv_layout="HND", sm_scale=scale, causal=is_causal).view(t, b, c).transpose(0, 1).contiguous()

    else:
        if use_sageattn:
            q = query.view(b, t, num_q_heads, head_dim)
            k = key.view(b, key.size(1), num_kv_heads, head_dim)
            v = value.view(b, value.size(1), num_kv_heads, head_dim)
            attn_output = sageattn_wrapper(q, k, v, scale)   # Hopper default
            attn_output = attn_output.view(b, t, c)

        else:
            q = query.view(query.size(0), query.size(1), num_q_heads, head_dim).transpose(1, 2)
            k = key.view(key.size(0), key.size(1), num_kv_heads, head_dim).transpose(1, 2)
            v = value.view(value.size(0), value.size(1), num_kv_heads, head_dim).transpose(1, 2)

            if pre_ampere and query.dtype==torch.bfloat16: #the Turing GPU does not support sdpa with bfloat16
                q = q.to(torch.float16)
                k = k.to(torch.float16)
                v = v.to(torch.float16)

            with sdpa_kernel(sdpa_backend):
                attn_output = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale
                )

            attn_output = attn_output.transpose(1, 2).view(b, t, c).contiguous()
    
    return attn_output


@kernel_registry.register('sdpa_sparse', 'cuda')
def sdpa_sparse_cuda(
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
    b, t, c = query.size()

    if use_spargeattn:
        q = query.view(b, t, num_q_heads, head_dim)
        k = key.view(b, key.size(1), num_kv_heads, head_dim)
        v = value.view(b, value.size(1), num_kv_heads, head_dim)

        attn_output = spargeattn_wrapper(
            q, k, v, 
            scale, sparse_mask
        )
        attn_output = attn_output.reshape(b, t, c)

    else:
        raise ImportError("Please install spas-sage-attn to use sparse sdpa.")

    return attn_output