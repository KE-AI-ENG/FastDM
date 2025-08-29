import torch
from fastdm.kernel.operators_set import gelu_and_mul
from fastdm.kernel.utils import set_global_backend, kernel_output_assert_close, benchmark_kernel

INPUT_ARGS = [
    [8192, 5120],
    [2048, 10240], 
]

def test_accuracy_gelu_and_mul(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        out_torch = gelu_and_mul(input)
        set_global_backend(backend)
        out_triton = gelu_and_mul(input)

        try:
            kernel_output_assert_close(out_torch, out_triton, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: M={M}, K={K}")
            print(f"{e}\n")

    print(f"test_accuracy_gelu_and_mul in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_gelu_and_mul(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_gelu_and_mul")
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        duration_torch, result = benchmark_kernel('gelu_and_mul', gelu_and_mul, 100, input)
        set_global_backend(backend)
        duration_backend, result = benchmark_kernel('gelu_and_mul', gelu_and_mul, 100, input)

        print(f"input_args[M={M}, K={K}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


if __name__ == "__main__":
    test_accuracy_gelu_and_mul(backend="triton")
    test_performance_gelu_and_mul(backend="triton")