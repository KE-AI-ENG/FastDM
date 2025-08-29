import torch
from fastdm.kernel.operators_set import quantize_to_int8, quantize_to_fp8
from fastdm.kernel.utils import set_global_backend, kernel_output_assert_close, benchmark_kernel

INPUT_ARGS = [
    [9216, 3072],
    [3072, 3072],
    [12288, 3072],
    [3072, 12288],
    [3072, 15360],
    [4096, 3072],
    [512, 3072],
    [4096, 12288],
    [512, 12288],
    [4608, 3072],
    [4608, 15360],
    [1280, 320],
    [1280, 1280],
    [1280, 2816],
    [640, 640],
    [1920, 640],
    [1280, 2048],
    [5120, 640],
    [640, 2560],
    [3840, 1280],
    [2560, 2048],
    [10240, 1280],
    [1280, 5120],
    [2, 320],
    [2, 1280],
    [2, 2816],
    [8192, 640],
    [154, 2048],
    [8192, 2560],
    [2048, 1280],
    [2048, 5120],
    [4608, 1536],
    [1536, 1536],
    [6144, 1536],
    [1536, 6144],
    [3072, 1536],
    [64, 1536],
    [8192, 1536],
    [1178, 1536],
    [8192, 6144],
    [1178, 6144],
    [2, 1536],
    [14, 3072],
    [14, 12288],
]

def test_accuracy_quant_per_token_int8_sym(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        x_q_torch, scales_torch, _ = quantize_to_int8(input, True)
        set_global_backend(backend)
        x_q_triton, scales_triton, _ = quantize_to_int8(input, True)

        try:
            kernel_output_assert_close(x_q_torch, x_q_triton, 0, 1)
            kernel_output_assert_close(scales_torch, scales_triton, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: M={M}, K={K}")
            print(f"{e}\n")

    print(f"test_accuracy_quant_per_token_int8_sym in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_quant_per_token_int8_sym(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_quant_per_token_int8_sym")
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        duration_torch, result = benchmark_kernel('quantize_to_int8', quantize_to_int8, 100, input, True)
        set_global_backend(backend)
        duration_backend, result = benchmark_kernel('quantize_to_int8', quantize_to_int8, 100, input, True)

        print(f"input_args[M={M},K={K}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


def test_accuracy_quant_per_token_int8_asym(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        x_q_torch, scales_torch, zeros_torch = quantize_to_int8(input, False)
        set_global_backend(backend)
        x_q_triton, scales_triton, zeros_triton = quantize_to_int8(input, False)

        try:
            kernel_output_assert_close(x_q_torch, x_q_triton, 0, 1)
            kernel_output_assert_close(scales_torch, scales_triton, None, None)
            kernel_output_assert_close(zeros_torch, zeros_triton, 0, 1)
        except Exception as e:
            unpass += 1
            print(f"ERROR: M={M}, K={K}")
            print(f"{e}\n")
    
    print(f"test_accuracy_quant_per_token_int8_asym in backend {backend}, test caces {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_quant_per_token_int8_asym(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_quant_per_token_int8_asym")
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        duration_torch, result = benchmark_kernel('quantize_to_int8', quantize_to_int8, 100, input, False)
        set_global_backend(backend)
        duration_backend, result = benchmark_kernel('quantize_to_int8', quantize_to_int8, 100, input, False)

        print(f"input_args[M={M},K={K}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


def test_accuracy_quant_per_token_fp8_sym(dtype = torch.bfloat16, backend="triton"):
    unpass = 0
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        x_q_torch, scales_torch = quantize_to_fp8(input)
        set_global_backend(backend)
        x_q_triton, scales_triton = quantize_to_fp8(input)

        try:
            kernel_output_assert_close(x_q_torch.to(torch.float), x_q_triton, rtol=1.6e-2, atol=2e-3)
            kernel_output_assert_close(scales_torch, scales_triton, None, None)
        except Exception as e:
            unpass += 1
            print(f"ERROR: M={M}, K={K}")
            print(f"{e}\n")

    print(f"test_accuracy_quant_per_token_fp8_sym in backend {backend}, test cases {len(INPUT_ARGS)-unpass}/{len(INPUT_ARGS)} are OK!")

def test_performance_quant_per_token_fp8_sym(dtype = torch.bfloat16, backend="triton"):
    print(f"test_performance_quant_per_token_fp8_sym")
    for (M, K) in INPUT_ARGS:
        input: torch.Tensor = torch.randn((M, K), device="cuda").to(dtype)
        set_global_backend("torch")
        duration_torch, result = benchmark_kernel('quantize_to_fp8', quantize_to_fp8, 100, input)
        set_global_backend(backend)
        duration_backend, result = benchmark_kernel('quantize_to_fp8', quantize_to_fp8, 100, input)

        print(f"input_args[M={M},K={K}], duration[torch]: {duration_torch * 1000} ms, duration[{backend}]: {duration_backend * 1000} ms")


if __name__ == "__main__":
    # int8_sym
    test_accuracy_quant_per_token_int8_sym(backend="triton")
    test_performance_quant_per_token_int8_sym(backend="triton")
    test_accuracy_quant_per_token_int8_sym(backend="cuda")
    test_performance_quant_per_token_int8_sym(backend="cuda")

    # int8_asym
    test_accuracy_quant_per_token_int8_asym(backend="triton")
    test_performance_quant_per_token_int8_asym(backend="triton")
    # test_accuracy_quant_per_token_int8_asym(backend="cuda")
    test_performance_quant_per_token_int8_asym(backend="cuda")

    # fp8_sym
    test_accuracy_quant_per_token_fp8_sym(backend="triton")
    test_performance_quant_per_token_fp8_sym(backend="triton")
    test_accuracy_quant_per_token_fp8_sym(backend="cuda")
    test_performance_quant_per_token_fp8_sym(backend="cuda")