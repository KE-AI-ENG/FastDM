# adapted from
# https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
# only works on post-Ampere GPUs right now

from typing import Optional

import torch
import os

import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:
    print("Please upgrade your triton version to 3.4.0 if you want to use the triton-backend attention kernel.")

from fastdm.kernel.registry import kernel_registry

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10

warp_specialize = True if is_blackwell() else False

def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    desc_k, desc_v,
                    off_b, off_h, H, N_KV: tl.constexpr, 
                    dtype: tl.constexpr, qk_scale,
                    BLOCK_N_KV: tl.constexpr, BLOCK_N_Q: tl.constexpr,
                    warp_specialize: tl.constexpr):
    lo, hi = 0, N_KV
    
    kv_offs_y_b = H * N_KV
    kv_offs_y_h = N_KV
    offset_bh = off_b * kv_offs_y_b + off_h * kv_offs_y_h
    # loop over k, v and update accumulator
    for start_n_kv in tl.range(lo, hi, BLOCK_N_KV, warp_specialize=warp_specialize):
        start_n_kv = tl.multiple_of(start_n_kv, BLOCK_N_KV)
        # -- compute qk ----
        kv_offset_y = offset_bh + start_n_kv
        k = desc_k.load([kv_offset_y, 0]).T
        v = desc_v.load([kv_offset_y, 0])
        qk = tl.dot(q, k)
        qk_s = qk * qk_scale
        if start_n_kv+BLOCK_N_KV >= hi: # just for (N_KV/BLOCK_N_KV!=0)
            mask_qk = (tl.arange(0, BLOCK_N_Q)[:, None] < BLOCK_N_Q) & ((start_n_kv + tl.arange(0, BLOCK_N_KV))[None, :] < hi)
            qk_s = tl.where(mask_qk, qk_s, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk_s, 1))
        qk_s = qk_s - m_ij[:, None]
        p = tl.math.exp(qk_s)               # exp(s_2 - m_2)
        # -- compute correction factor
        alpha = tl.math.exp(m_i - m_ij)     # exp(m_1 - m_2)
        l_ij = tl.sum(p, 1)                 # rowsum(exp(s_2 - m_2)) 
        # -- update output accumulator --
        acc = acc * alpha[:, None]          # exp(m_1 - m_2)*O_1~
        # -- prepare p and v for the dot
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)             # exp(m_1 - m_2)*O_1~ + exp(m_1 - m_2)@V
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_N_Q = nargs["BLOCK_N_Q"]
    BLOCK_N_KV = nargs["BLOCK_N_KV"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_N_Q, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N_KV, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N_KV, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_N_Q, HEAD_DIM]

NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config(dict(BLOCK_N_Q=64, BLOCK_N_KV=32), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook) # TODO support autotune
    # triton.Config({'BLOCK_N_Q': BM, 'BLOCK_N_KV': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \   
    # for BM in [64, 128]\
    # for BN in [32, 64, 128]\
    # for s in NUM_STAGES_OPTIONS \
    # for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(dict(BLOCK_N_Q=128, BLOCK_N_KV=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_N_Q = conf.kwargs["BLOCK_N_Q"]
    BLOCK_N_KV = conf.kwargs["BLOCK_N_KV"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_N_Q * BLOCK_N_KV < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_Q = kwargs["N_Q"]

    # Filter out configs where BLOCK_N_Q > N_Q
    return [conf for conf in configs if conf.kwargs.get("BLOCK_N_Q", 0) <= N_Q]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(
                configs=list(filter(keep, configs)), 
                # configs=configs, 
                key=["N_Q", "HEAD_DIM", "warp_specialize"],
                prune_configs_by={'early_config_prune': prune_invalid_configs}
                )
@triton.jit
def _attn_fwd(sm_scale,
              B, H, 
              desc_q, desc_k, desc_v, desc_o, 
              N_Q, N_KV, 
              HEAD_DIM: tl.constexpr,
              warp_specialize: tl.constexpr,
              BLOCK_N_Q: tl.constexpr, 
              BLOCK_N_KV: tl.constexpr,
              ):
    dtype = tl.bfloat16 
    tl.static_assert(BLOCK_N_KV <= HEAD_DIM)

    qo_y_size = B * H * N_Q
    kv_y_size = B * H * N_KV
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[qo_y_size, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N_Q, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[kv_y_size, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N_KV, HEAD_DIM])
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[kv_y_size, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N_KV, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[qo_y_size, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N_Q, HEAD_DIM])

    # load q: it will stay in SRAM throughout
    start_bh = tl.program_id(0)
    start_n_q = tl.program_id(1)
    off_b = start_bh // H
    off_h = start_bh % H
    qo_offs_y_b = H * N_Q
    qo_offs_y_h = N_Q
    offset_bh = off_b * qo_offs_y_b + off_h * qo_offs_y_h
    qo_offset_y = offset_bh + start_n_q * BLOCK_N_Q
    q = desc_q.load([qo_offset_y, 0])

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_N_Q], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_N_Q], dtype=tl.float32)
    acc = tl.zeros([BLOCK_N_Q, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,
                                    desc_k, desc_v,
                                    off_b, off_h, H, N_KV, 
                                    dtype, qk_scale,
                                    BLOCK_N_KV, BLOCK_N_Q,
                                    warp_specialize)

    # epilogue
    m_i += tl.math.log(l_i)
    acc = acc / l_i[:, None]

    if start_n_q * BLOCK_N_Q + BLOCK_N_Q >= N_Q:    # just for ((N_Q/BLOCK_N_Q!=0) and (B!=0 or H!=0))
        mask_o = (start_n_q * BLOCK_N_Q + tl.arange(0, BLOCK_N_Q)[:, None] < N_Q) & (tl.arange(0, HEAD_DIM)[None, :] < HEAD_DIM)
        acc = tl.where(mask_o, acc, 0.0)
    desc_o.atomic_add([qo_offset_y, 0], acc.to(dtype))  # [250905] atomic_add can't support debug

    
@kernel_registry.register('sdpa', 'triton')
def sdpa_triton(
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
    q = query.view(query.size(0), query.size(1), num_q_heads, head_dim).transpose(1, 2).contiguous()
    k = key.view(key.size(0), key.size(1), num_kv_heads, head_dim).transpose(1, 2).contiguous()
    v = value.view(value.size(0), value.size(1), num_kv_heads, head_dim).transpose(1, 2).contiguous()

    # shape constraints
    assert q.shape[1] == v.shape[1] and q.shape[1] == v.shape[1]        # assert num_heads, only support MHA now
    assert k.shape[2] == v.shape[2]                                     # assert seq_len_kv
    assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]    # assert head_dim
    assert q.shape[-1] in {16, 32, 64, 128, 256}
    batch_size = q.shape[0]
    num_head_q = q.shape[1]
    num_head_kv = k.shape[1]
    seq_len_q = q.shape[2]
    seq_len_kv = k.shape[2]

    o = torch.zeros_like(q)
    extra_kern_args = {}

    # Use device_descriptor for Hopper + warpspec.
    if supports_host_descriptor() and not (is_hopper() and warp_specialize):
        # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
        dummy_block = [1, 1]
        desc_q = TensorDescriptor(q, shape=[batch_size*num_head_q*seq_len_q, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
        desc_v = TensorDescriptor(v, shape=[batch_size*num_head_kv*seq_len_kv, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
        desc_k = TensorDescriptor(k, shape=[batch_size*num_head_kv*seq_len_kv, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
        desc_o = TensorDescriptor(o, shape=[batch_size*num_head_q*seq_len_q, head_dim], strides=[head_dim, 1], block_shape=dummy_block)
    else:
        desc_q = q
        desc_v = v
        desc_k = k
        desc_o = o

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)

    def grid(META):
        return (batch_size*num_head_q, triton.cdiv(seq_len_q, META["BLOCK_N_Q"]))

    if is_blackwell() and warp_specialize:
        if head_dim == 128 and q.dtype == torch.float16:
            extra_kern_args["maxnreg"] = 168
        else:
            extra_kern_args["maxnreg"] = 80

    _attn_fwd[grid](
        scale, 
        batch_size, num_head_q,
        desc_q, desc_k, desc_v, desc_o,
        N_Q=seq_len_q, N_KV = seq_len_kv, HEAD_DIM=head_dim,
        warp_specialize=warp_specialize,
        **extra_kern_args)

    attn_output = o.transpose(1, 2).contiguous().view(b, t, c)
    return attn_output


@kernel_registry.register('sdpa_sparse', 'triton')
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
    raise ValueError(f"Now sparge_attn isn't supported in kernel backend triton.")