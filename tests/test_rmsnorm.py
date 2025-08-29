import torch
from fastdm.kernel.operators_set import rms_norm
from fastdm.kernel.utils import set_global_backend, kernel_output_assert_close, benchmark_kernel

INPUT_ARGS = [
    [1, 4096, 24, 128],
    [1, 512, 24, 128],
    [1, 4608, 24, 128],
    [1, 14, 24, 128],
    [2, 4096, 24, 64],
    [2, 589, 24, 64],
]

def test_accuracy_rmsnorm(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (B, S, head_num, head_dim) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((B, S, head_num, head_dim), device="cuda").to(dtype)
        scale: torch.Tensor = torch.randn((head_dim,), device="cuda").to(dtype)
        eps: float = 1e-06
        set_global_backend("torch")
        C_torch = rms_norm(input, scale, eps)
        set_global_backend(backend)
        C_triton = rms_norm(input, scale, eps)

        try:
            kernel_output_assert_close(C_torch, C_triton, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: B={B}, S={S}, head_num={head_num}, head_dim={head_dim}")
            print(f"{e}\n")
    
    for (B, S, head_num, head_dim) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((B, S, head_num, head_dim), device="cuda").to(dtype)
        scale: torch.Tensor = torch.randn((head_num * head_dim,), device="cuda").to(dtype)
        eps: float = 1e-06
        set_global_backend(backend)
        res1 = rms_norm(input, scale, eps)
        scale = scale.view(head_num, head_dim)
        set_global_backend(backend)
        res2 = rms_norm(input, scale, eps)

        try:
            kernel_output_assert_close(res1, res2, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: B={B}, S={S}, head_num={head_num}, head_dim={head_dim}")
            print(f"{e}\n")

    print(f"test_accuracy_rmsnorm in backend {backend}, test cases {2*len(INPUT_ARGS)-unpass}/{2*len(INPUT_ARGS)} are OK!")

def test_performance_rmsnorm(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_rmsnorm")
    for (B, S, head_num, head_dim) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((B, S, head_num, head_dim), device="cuda").to(dtype)
        scale: torch.Tensor = torch.randn((head_dim,), device="cuda").to(dtype)
        eps: float = 1e-06
        set_global_backend("torch")
        duration_torch, result = benchmark_kernel('rms_norm', rms_norm, 100, input, scale, eps)
        set_global_backend(backend)
        duration_backend, result = benchmark_kernel('rms_norm', rms_norm, 100, input, scale, eps)

        print(f"input_args[B={B},S={S},head_num={head_num},head_dim={head_dim}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


if __name__ == "__main__":
    test_accuracy_rmsnorm(backend="triton")
    test_performance_rmsnorm(backend="triton")
    test_accuracy_rmsnorm(backend="cuda")
    test_performance_rmsnorm(backend="cuda")