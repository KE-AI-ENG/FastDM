import torch
from fastdm.kernel.operators_set import rotary_pos_embedding
from fastdm.kernel.utils import set_global_backend, benchmark_kernel

INPUT_ARGS = [
    [1, 4608, 3072, 128],
]

def test_accuracy_rope(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (B, S, hidden_dim, head_size) in INPUT_ARGS:
        max_seq_len = S
        query_torch: torch.Tensor = torch.randn((B, S, hidden_dim), device="cuda").to(dtype)
        key_torch: torch.Tensor = torch.randn((B, S, hidden_dim), device="cuda").to(dtype)
        cos_sin_cache: torch.Tensor = torch.rand((max_seq_len, head_size), device="cuda").to(dtype)
        is_neox: bool = False

        query_triton = torch.empty_like(query_torch).copy_(query_torch)
        key_triton = torch.empty_like(key_torch).copy_(key_torch)

        set_global_backend("torch")
        rotary_pos_embedding(query_torch, key_torch, head_size, cos_sin_cache, is_neox)
        set_global_backend(backend)
        rotary_pos_embedding(query_triton, key_triton, head_size, cos_sin_cache, is_neox)

        try:
            assert (abs(query_torch - query_triton) < 1.6e-2*abs(query_torch) + 0.0079).all().item()
            assert (abs(key_torch - key_triton) < 1.6e-2*abs(key_torch) + 0.0079).all().item()
        except Exception as e:
            unpass += 1
            print(f"ERROR: B={B}, S={S}, hidden_dim={hidden_dim}, head_size={head_size}")
            print(f"{e}\n")

    print(f"test_accuracy_rope in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_rope(dtype = torch.bfloat16, backend1="cuda", backend2="triton"):
    print(f"test_performance_rope")
    for (B, S, hidden_dim, head_size) in INPUT_ARGS:
        max_seq_len = S
        query_torch: torch.Tensor = torch.randn((B, S, hidden_dim), device="cuda").to(dtype)
        key_torch: torch.Tensor = torch.randn((B, S, hidden_dim), device="cuda").to(dtype)
        cos_sin_cache: torch.Tensor = torch.rand((max_seq_len, head_size), device="cuda").to(dtype)
        is_neox: bool = False

        query_triton = torch.empty_like(query_torch).copy_(query_torch)
        key_triton = torch.empty_like(key_torch).copy_(key_torch)

        set_global_backend("torch")
        duration_backend1, result = benchmark_kernel('rotary_pos_embedding', rotary_pos_embedding, 100, query_torch, key_torch, head_size, cos_sin_cache, is_neox)
        set_global_backend(backend2)
        duration_backend2, result = benchmark_kernel('rotary_pos_embedding', rotary_pos_embedding, 100, query_triton, key_triton, head_size, cos_sin_cache, is_neox)

        performance = (duration_backend1 - duration_backend2) / duration_backend1 * 100
        print(f"input_args[B={B}, S={S}, hidden_dim={hidden_dim}, head_size={head_size}], duration[{backend1}]: {duration_backend1 * 1000} ms, duration[{backend2}]: {duration_backend2 * 1000} ms, ({backend1}-{backend2})/{backend1}={performance}%")


if __name__ == "__main__":
    test_accuracy_rope(backend="triton")
    test_accuracy_rope(backend="cuda")
    test_performance_rope(backend1="cuda", backend2="triton")