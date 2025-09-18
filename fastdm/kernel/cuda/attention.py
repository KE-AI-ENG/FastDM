from typing import Optional

import torch

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

from fastdm.kernel.registry import kernel_registry

import fastdm.cuda_ops as cuda_ops

def sageattn_wrapper(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        scale: float
    ) -> torch.Tensor:
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
    fp8_attn_: bool=False,
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
        if torch.cuda.get_device_capability()[0]>=9 and fp8_attn_:
            q = query.view(query.size(0), query.size(1), num_q_heads, head_dim).to(torch.float8_e4m3fn)
            k = key.view(key.size(0), key.size(1), num_kv_heads, head_dim).to(torch.float8_e4m3fn)
            v = value.view(value.size(0), value.size(1), num_kv_heads, head_dim).to(torch.float8_e4m3fn)

            attn_output = cuda_ops.flash_attention_fp8_fwd_(q, k, v, scale, is_causal)

            attn_output = attn_output.view(b, t, c).contiguous()
        else:
            if use_sageattn:
                q = query.view(b, t, num_q_heads, head_dim)
                k = key.view(b, key.size(1), num_kv_heads, head_dim)
                v = value.view(b, value.size(1), num_kv_heads, head_dim)
                attn_output = sageattn_wrapper(q, k, v, scale)
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
                    attn_output = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, dropout_p=0.0, is_causal=is_causal, scale=scale
                    )

                attn_output = attn_output.transpose(1, 2).view(b, t, c).contiguous()
    
    return attn_output