import torch
from fastdm.kernel.operators_set import scaled_dot_product_attention
from fastdm.kernel.utils import set_global_backend, kernel_output_assert_close, benchmark_kernel

INPUT_ARGS = [
    [2, 8704, 24, 128], #flux
    [2, 4685, 24, 64], #sd3.5-medium
    # [1, 75600, 40, 128], #wan2.2
]

def test_accuracy_sdpa(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (B, S, head_num, head_dim) in INPUT_ARGS:
        query: torch.Tensor = torch.randn((B, S, head_num*head_dim), device="cuda").to(dtype)
        key: torch.Tensor = torch.randn((B, S, head_num*head_dim), device="cuda").to(dtype)
        value: torch.Tensor = torch.randn((B, S, head_num*head_dim), device="cuda").to(dtype)
        scale: float = 1.0 / (head_dim ** 0.5)
        set_global_backend("torch")
        C_torch = scaled_dot_product_attention(query, key, value, head_num, head_num, head_dim, scale=scale)
        set_global_backend(backend)
        C_backend = scaled_dot_product_attention(query, key, value, head_num, head_num, head_dim, scale=scale)

        try:
            kernel_output_assert_close(C_torch, C_backend, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: B={B}, S={S}, head_num={head_num}, head_dim={head_dim}")
            print(f"{e}\n")

    print(f"test_accuracy_sdpa in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_sdpa(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_sdpa")
    for (B, S, head_num, head_dim) in INPUT_ARGS:
        query: torch.Tensor = torch.randn((B, S, head_num*head_dim), device="cuda").to(dtype)
        key: torch.Tensor = torch.randn((B, S, head_num*head_dim), device="cuda").to(dtype)
        value: torch.Tensor = torch.randn((B, S, head_num*head_dim), device="cuda").to(dtype)
        scale: float = 1.0 / (head_dim ** 0.5)
        set_global_backend("torch")
        duration_torch, _ = benchmark_kernel('sdpa', scaled_dot_product_attention, 100, query, key, value, head_num, head_num, head_dim, False, scale, False)
        set_global_backend(backend)
        duration_backend, _ = benchmark_kernel('sdpa', scaled_dot_product_attention, 100, query, key, value, head_num, head_num, head_dim, False, scale, False)

        print(f"input_args[B={B},S={S},head_num={head_num},head_dim={head_dim}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


if __name__ == "__main__":
    # test_accuracy_sdpa(backend="cuda")
    # test_performance_sdpa(backend="cuda")
    test_accuracy_sdpa(backend="triton")
    test_performance_sdpa(backend="triton")