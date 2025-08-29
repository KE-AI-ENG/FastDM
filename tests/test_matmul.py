import torch
from fastdm.kernel.operators_set import int8_matmul, fp8_matmul
from fastdm.kernel.utils import set_global_backend, kernel_output_assert_close, benchmark_kernel

INPUT_ARGS = [
    [4096, 3072, 9216],
    [512, 3072, 9216],
    [512, 3072, 3072],
    [4096, 3072, 3072],
    [4096, 3072, 12288],
    [4096, 12288, 3072],
    [512, 3072, 12288],
    [512, 12288, 3072],
    [4608, 3072, 12288],
    [4608, 3072, 9216],
    [4608, 15360, 3072],
    [14, 3072, 9216],
    [14, 3072, 3072],
    [14, 3072, 12288],
    [14, 12288, 3072],
    [8192, 1536, 4608],
    [1178, 1536, 4608],
    [1178, 1536, 1536],
    [8192, 1536, 1536],
    [8192, 1536, 6144],
    [8192, 6144, 1536],
    [1178, 1536, 6144],
    [1178, 6144, 1536],
    [2, 1536, 3072],
    [8192, 1536, 64],
    [2, 320, 1280],
    [2, 1280, 1280],
    [2, 2816, 1280],
    [8192, 640, 640],
    [8192, 640, 1920],
    [154, 2048, 1280],
    [8192, 640, 5120],
    [8192, 2560, 640],
    [2048, 1280, 1280],
    [2048, 1280, 3840],
    [154, 2048, 2560],
    [2048, 1280, 10240],
    [2048, 5120, 1280],
]

def test_accuracy_matmul_int8(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (M, K, N) in INPUT_ARGS:
        a: torch.Tensor = torch.randint(-128, 128, (M, K), device="cuda").to(torch.int8)
        b: torch.Tensor = torch.randint(-128, 128, (K, N), device="cuda").to(torch.int8)
        b = b.t().contiguous().t()  # just for cuda
        scale_a: torch.Tensor = torch.randn((M, 1), device="cuda").to(torch.float32)
        scale_b: torch.Tensor = torch.randn((N, 1), device="cuda").to(torch.float32)
        out_dtype: torch.dtype = torch.bfloat16
        azp_adj: torch.Tensor = torch.randint(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max, (1, N), device="cuda").to(torch.int32)
        azp: torch.Tensor = torch.randint(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max, (M, 1), device="cuda").to(torch.int32)
        bias = torch.randn(N, device="cuda").to(torch.bfloat16)

        set_global_backend("torch")
        c_torch = int8_matmul(a, b, scale_a, scale_b, out_dtype, azp_adj, azp, bias)
        set_global_backend(backend)
        c_triton = int8_matmul(a, b, scale_a, scale_b, out_dtype, azp_adj, azp, bias)

        try:
            kernel_output_assert_close(c_torch, c_triton, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: M={M}, K={K}, N={N}")
            print(f"{e}\n")
    
    print(f"test_accuracy_matmul_int8 in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_matmul_int8(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_matmul_int8")
    for (M, K, N) in INPUT_ARGS:
        a: torch.Tensor = torch.randint(-128, 128, (M, K), device="cuda").to(torch.int8)
        b: torch.Tensor = torch.randint(-128, 128, (K, N), device="cuda").to(torch.int8)
        b = b.t().contiguous().t()  # just for cuda
        scale_a: torch.Tensor = torch.randn((M, 1), device="cuda").to(torch.float32)
        scale_b: torch.Tensor = torch.randn((N, 1), device="cuda").to(torch.float32)
        out_dtype: torch.dtype = torch.bfloat16
        azp_adj: torch.Tensor = torch.randint(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max, (1, N), device="cuda").to(torch.int32)
        azp: torch.Tensor = torch.randint(torch.iinfo(torch.int8).min, torch.iinfo(torch.int8).max, (M, 1), device="cuda").to(torch.int32)
        bias = torch.randn(N, device="cuda").to(torch.bfloat16)

        set_global_backend("torch")
        duration_torch, result = benchmark_kernel('int8_matmul', int8_matmul, 100, a, b, scale_a, scale_b, out_dtype, azp_adj, azp, bias)
        set_global_backend(backend)
        duration_backend, result = benchmark_kernel('int8_matmul', int8_matmul, 100, a, b, scale_a, scale_b, out_dtype, azp_adj, azp, bias)

        print(f"input_args[M={M},K={K},N={N}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


def test_accuracy_matmul_fp8(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (M, K, N) in INPUT_ARGS:
        a: torch.Tensor = torch.randn((M, K), device="cuda").to(torch.float8_e4m3fn)
        b: torch.Tensor = torch.randn((K, N), device="cuda").to(torch.float8_e4m3fn)
        b = b.t().contiguous().t()
        scale_a: torch.Tensor = torch.randn((M, 1), device="cuda").to(torch.float32)
        scale_b: torch.Tensor = torch.randn((N, 1), device="cuda").to(torch.float32)
        out_dtype: torch.dtype = torch.bfloat16
        bias = torch.randn(N, device="cuda").to(torch.bfloat16)

        set_global_backend("torch")
        c_torch = fp8_matmul(a, b, scale_a, scale_b, out_dtype, bias)
        set_global_backend(backend)
        c_triton = fp8_matmul(a, b, scale_a, scale_b, out_dtype, bias)

        try:
            kernel_output_assert_close(c_torch, c_triton, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: M={M}, K={K}, N={N}")
            print(f"{e}\n")

    print(f"test_accuracy_quant_per_token_fp8_sym in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_matmul_fp8(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_matmul_fp8")
    for (M, K, N) in INPUT_ARGS:
        a: torch.Tensor = torch.randn((M, K), device="cuda").to(torch.float8_e4m3fn)
        b: torch.Tensor = torch.randn((K, N), device="cuda").to(torch.float8_e4m3fn)
        b = b.t().contiguous().t()
        scale_a: torch.Tensor = torch.randn((M, 1), device="cuda").to(torch.float32)
        scale_b: torch.Tensor = torch.randn((N, 1), device="cuda").to(torch.float32)
        out_dtype: torch.dtype = torch.bfloat16
        bias = torch.randn(N, device="cuda").to(torch.bfloat16)

        set_global_backend("torch")
        duration_torch, result = benchmark_kernel('fp8_matmul', fp8_matmul, 100, a, b, scale_a, scale_b, out_dtype, bias)
        set_global_backend(backend)
        duration_backend, result = benchmark_kernel('fp8_matmul', fp8_matmul, 100, a, b, scale_a, scale_b, out_dtype, bias)

        print(f"input_args[M={M},K={K},N={N}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


if __name__ == "__main__":
    # int8_asym
    test_accuracy_matmul_int8(backend="triton")
    test_performance_matmul_int8(backend="triton")
    # test_accuracy_matmul_int8(backend="cuda")
    test_performance_matmul_int8(backend="cuda")

    # fp8_mm
    test_accuracy_matmul_fp8(backend="triton")
    test_performance_matmul_fp8(backend="triton")
    # test_accuracy_matmul_fp8(backend="cuda")
    test_performance_matmul_fp8(backend="cuda")