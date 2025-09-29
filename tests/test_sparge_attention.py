import torch
from fastdm.kernel.operators_set import scaled_dot_product_attention, sparse_scaled_dot_product_attention
from fastdm.kernel.utils import set_global_backend, kernel_output_assert_close, benchmark_kernel
from einops import repeat
import math

INPUT_ARGS = [
    # [1, 4608, 1, 4608, 24, 24, 128, 'False'],       # flux_fp8_bf16
    # [1, 4110, 1, 4110, 24, 24, 128, 'False'],      # qwen_fp8
    # [2, 4096, 2, 4096, 10, 10, 64, 'False'],
    # # [2, 4096, 2, 77, 10, 10, 64, 'False'],         # sdxl_bf16    # TODO
    # [2, 1024, 2, 1024, 20, 20, 64, 'False'],
    # # [2, 1024, 2, 77, 20, 20, 64, 'False'],         # sdxl_bf16
    # [1, 4106, 1, 4106, 24, 24, 128, 'False'],      # qwen_bf16
    # [2, 4685, 2, 4685, 24, 24, 64, 'False'],       # sd3_fp8
    # [2, 4096, 2, 4096, 24, 24, 64, 'False'],       # sd3_fp8

    [1, 489830, 1, 489830, 40, 40, 128, 'False'],  # Wan2.2-I2V-A14B
    [1, 489830, 1, 512, 40, 40, 128, 'False'],     # Wan2.2-I2V-A14B
]

def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p=0.0,
    upcast=True,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        dropout_p: float
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    # d = q.shape[-1]
    # scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d)
    b, t_q, h, d = q.shape
    _, t_kv, _, _ = k.shape
    q_reshaped = q.transpose(1,2).reshape(b * h, t_q, d)    # (bh)td
    k_reshaped = k.transpose(1,2).reshape(b * h, t_kv, d)   # (bh)sd
    scores = torch.bmm(q_reshaped,k_reshaped.transpose(1,2)) / math.sqrt(d) # (bh)ts

    # attention = torch.softmax(scores, dim=-1).to(v.dtype)
    attention = torch.softmax(scores, dim=-1)
    dropout_scaling = 1.0 / (1 - dropout_p)
    attention_drop = attention
    # output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    v_reshaped = v.transpose(1,2).reshape(b * h, t_kv, d)   # (bh)sd
    output = torch.bmm(attention_drop, v_reshaped * dropout_scaling)    # (bh)td
    output = output.reshape(b, h, t_q, d).transpose(1,2)    # bhtd

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


if torch.cuda.get_device_capability() == (9,0):
    BLOCK_Q = 64
    BLOCK_K = 128
else:
    BLOCK_Q = 128
    BLOCK_K = 64 

def test_accuracy_sdpa(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (B_q, S_q, B_kv, S_kv, head_num_q, head_num_kv, head_dim, is_causal) in INPUT_ARGS:
        torch.manual_seed(0)
        query: torch.Tensor = torch.randn((B_q, S_q, head_num_q*head_dim), device="cuda").to(dtype)
        key: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        value: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        scale: float = 1.0 / (head_dim ** 0.5)
        mask_id = torch.ones(B_q, head_num_q, S_q//BLOCK_Q, S_kv//BLOCK_K).to("cuda")
        
        total_token_num = B_q*S_q*head_num_q
        token_num_per_kernel = 1*489830*34*(128//head_dim)
        if (head_dim in [64,128]) and (total_token_num >= token_num_per_kernel): # if seq_len is extremely large
            set_global_backend("torch")
            O_torch = scaled_dot_product_attention(query, key, value, head_num_q, head_num_kv, head_dim, scale=scale)
        else:
            O_torch, _ = attention_ref(
                query.view(B_q, S_q, head_num_q, head_dim),
                key.view(B_kv, S_kv, head_num_kv, head_dim),
                value.view(B_kv, S_kv, head_num_kv, head_dim),
                dropout_p=0.0,
                upcast=True,
            )
            O_torch = O_torch.reshape(B_q, S_q, head_num_q*head_dim)

        set_global_backend(backend)
        O_backend = sparse_scaled_dot_product_attention(query, key, value, 
                                                 head_num_q, head_num_kv, head_dim, scale=scale,
                                                 sparse_mask=mask_id)

        try:
            kernel_output_assert_close(O_torch, O_backend, rtol=0.0, atol=1.8e-2)   # rtol=0.0, atol=1.8e-2
        except Exception as e:
            unpass += 1
            print(f"ERROR: B_q={B_q}, S_q={S_q}, B_kv={B_kv}, S_kv={S_kv}, head_num={head_num_q}, head_dim={head_dim}")
            print(f"{e}\n")

    print(f"test_accuracy_sdpa in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_sdpa(dtype = torch.bfloat16, backend1="cuda", backend2="triton"):
    print(f"test_performance_sdpa")
    for (B_q, S_q, B_kv, S_kv, head_num_q, head_num_kv, head_dim, is_causal) in INPUT_ARGS:
        query: torch.Tensor = torch.randn((B_q, S_q, head_num_q*head_dim), device="cuda").to(dtype)
        key: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        value: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        scale: float = 1.0 / (head_dim ** 0.5)
        mask_id = torch.ones(B_q, head_num_q, S_q//BLOCK_Q, S_kv//BLOCK_K).to("cuda")
        set_global_backend(backend1)
        duration_backend1, _ = benchmark_kernel('sdpa', sparse_scaled_dot_product_attention, 100, 
                                                query, key, value, 
                                                head_num_q, head_num_kv, head_dim, 
                                                False, scale, mask_id)
        set_global_backend(backend2)
        duration_backend2, _ = benchmark_kernel('sdpa', sparse_scaled_dot_product_attention, 100, 
                                                query, key, value, 
                                                head_num_q, head_num_kv, head_dim, 
                                                False, scale, mask_id)
        
        performance = (duration_backend1 - duration_backend2) / duration_backend1 * 100
        print(f"input_args[B={B_q},S_q={S_q},S_kv={S_kv},head_num={head_num_q},head_dim={head_dim}], duration[{backend1}]: {duration_backend1 * 1000} ms, duration[{backend2}]: {duration_backend2 * 1000} ms, ({backend1}-{backend2})/{backend1}={performance}%")


if __name__ == "__main__":
    test_accuracy_sdpa(backend="cuda")
    # test_performance_sdpa(backend1="cuda", backend2="cuda") # should use the mask_id in model