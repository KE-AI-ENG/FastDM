import torch
from fastdm.kernel.operators_set import scaled_dot_product_attention
from fastdm.kernel.utils import set_global_backend, kernel_output_assert_close, benchmark_kernel
from einops import repeat
import math

INPUT_ARGS = [
    # [1, 75600, 40, 128], #wan2.2
    # [1, 4608, 1, 4608, 24, 24, 128, 'False', 'True'],    # flux_fp8   # TODO
    [1, 4110, 1, 4110, 24, 24, 128, 'False', 'False'],      # qwen_fp8
    # [2, 4685, 2, 4685, 24, 24, 64, 'False', 'True'],     # sd3_fp8
    # [2, 4096, 2, 4096, 24, 24, 64, 'False', 'True'],     # sd3_fp8
    [2, 4096, 2, 4096, 10, 10, 64, 'False', 'False'],
    [2, 4096, 2, 77, 10, 10, 64, 'False', 'False'],         # sdxl_bf16
    [2, 1024, 2, 1024, 20, 20, 64, 'False', 'False'],
    [2, 1024, 2, 77, 20, 20, 64, 'False', 'False'],         # sdxl_bf16
    [1, 4106, 1, 4106, 24, 24, 128, 'False', 'False'],      # qwen_bf16
    [2, 4685, 2, 4685, 24, 24, 64, 'False', 'False'],       # sd3_fp8
    [2, 4096, 2, 4096, 24, 24, 64, 'False', 'False'],       # sd3_fp8
]

def attention_ref(
    q,
    k,
    v,
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
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(d)
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    dropout_scaling = 1.0 / (1 - dropout_p)
    attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def test_accuracy_sdpa(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (B_q, S_q, B_kv, S_kv, head_num_q, head_num_kv, head_dim, is_causal, fp8_attn_) in INPUT_ARGS:
        torch.manual_seed(0)
        query: torch.Tensor = torch.randn((B_q, S_q, head_num_q*head_dim), device="cuda").to(dtype)
        key: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        value: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        scale: float = 1.0 / (head_dim ** 0.5)

        C_torch, _ = attention_ref(
                query.view(B_q, S_q, head_num_q, head_dim),
                key.view(B_kv, S_kv, head_num_kv, head_dim),
                value.view(B_kv, S_kv, head_num_kv, head_dim),
                dropout_p=0.0,
                upcast=True,
            )
        set_global_backend(backend)
        C_backend = scaled_dot_product_attention(query, key, value, head_num_q, head_num_kv, head_dim, scale=scale)

        try:
            kernel_output_assert_close(C_torch.reshape((B_q, S_q, head_num_q*head_dim)), C_backend, rtol=0.0, atol=1.8e-2)   # rtol=0.0, atol=1.8e-2
        except Exception as e:
            unpass += 1
            print(f"ERROR: B_q={B_q}, S_q={S_q}, B_kv={B_kv}, S_kv={S_kv}, head_num={head_num_q}, head_dim={head_dim}")
            print(f"{e}\n")

    print(f"test_accuracy_sdpa in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_sdpa(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_sdpa")
    for (B_q, S_q, B_kv, S_kv, head_num_q, head_num_kv, head_dim, is_causal, fp8_attn_) in INPUT_ARGS:
        query: torch.Tensor = torch.randn((B_q, S_q, head_num_q*head_dim), device="cuda").to(dtype)
        key: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        value: torch.Tensor = torch.randn((B_kv, S_kv, head_num_kv*head_dim), device="cuda").to(dtype)
        scale: float = 1.0 / (head_dim ** 0.5)
        set_global_backend("torch")
        duration_torch, _ = benchmark_kernel('sdpa', scaled_dot_product_attention, 100, query, key, value, head_num_q, head_num_kv, head_dim, False, scale, False)
        set_global_backend(backend)
        duration_backend, _ = benchmark_kernel('sdpa', scaled_dot_product_attention, 100, query, key, value, head_num_q, head_num_kv, head_dim, False, scale, False)

        print(f"input_args[B={B_q},S_q={S_q},S_kv={S_kv},head_num={head_num_q},head_dim={head_dim}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


if __name__ == "__main__":
    # test_accuracy_sdpa(backend="cuda")
    test_performance_sdpa(backend="cuda")
    test_accuracy_sdpa(backend="triton")
    test_performance_sdpa(backend="triton")