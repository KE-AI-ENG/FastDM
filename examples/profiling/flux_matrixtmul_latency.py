import time

import torch

from fastdm.kernel.operators_set import fp8_matmul, int8_matmul

from fastdm.kernel.utils import set_global_backend

set_global_backend("cuda")

iter_num = 100
warmups = 2

output_dtype = torch.bfloat16

def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def rand_int8(shape: tuple, device: str = "cuda"):
    return to_int8(torch.rand(shape, device=device) * 255 - 128)

def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)

def test_fastdm_int8(m: int, n: int, k: int, out_dtype: torch.dtype, use_bias: bool, azp_per_token: bool): 
    m_azp = m if azp_per_token else 1
    scale_a = torch.randn((m_azp, 1), device="cuda", dtype=torch.float32) / 10
    scale_b = torch.randn((1, n), device="cuda", dtype=torch.float32) / 10

    aq_i8 = rand_int8((m, k))

    bq_i8 = rand_int8((n, k)).t()
    bq_i32 = bq_i8.to(dtype=torch.int32)

    azp_a = torch.rand(
        (m_azp, 1), device="cuda", dtype=torch.float32) * 10 + 1.5
    azp_aq_i8 = (azp_a / scale_a).to(dtype=torch.int8)
    azp_a = azp_aq_i8.to(dtype=torch.float32) * scale_a  # correct for rounding

    if use_bias:
        bias = torch.rand((1, n), device="cuda", dtype=out_dtype) * 10 + 2.5
    else:
        bias = torch.zeros((1, n), device="cuda", dtype=out_dtype)

    # Hadamard is just the sum of the cols
    azp_adj_i32 = bq_i32.sum(dim=0, keepdim=True, dtype=torch.int32)
    azp_i32 = azp_aq_i8.to(dtype=torch.int32)
    func_bias = bias if use_bias else None

    #warm up
    for i in range(warmups):
        out = int8_matmul(aq_i8, bq_i8, scale_a, scale_b,
                                            out_dtype, azp_adj_i32, azp_i32,
                                            func_bias)
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(iter_num):
        out = int8_matmul(aq_i8, bq_i8, scale_a, scale_b,
                                            out_dtype, azp_adj_i32, azp_i32,
                                            func_bias)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time-start_time)/iter_num*1000

    return avg_time

def test_fastdm_fp8(m: int, n: int, k: int, use_bias: bool, out_dtype: type[torch.dtype] = output_dtype):
    a = to_fp8(torch.randn((m, k), device="cuda"))
    b = to_fp8(torch.randn((n, k), device="cuda").t())

    scale_a_ = torch.randn((m,), device="cuda", dtype=torch.float32) * 0.001
    scale_b_ = torch.randn((n,), device="cuda", dtype=torch.float32) * 0.001

    if use_bias:
        bias = torch.rand((n, ), device="cuda", dtype=out_dtype) * 10
    else:
        bias = None

    #warm up
    for i in range(warmups):
        out = fp8_matmul(a, b, scale_a_, scale_b_, out_dtype, bias)
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(iter_num):
        out = fp8_matmul(a, b, scale_a_, scale_b_, out_dtype, bias)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time-start_time)/iter_num*1000

    return avg_time

def test_torch_mm(m: int, n: int, k: int, out_dtype: torch.dtype, use_bias: bool):
    a = torch.rand((m, k), dtype=out_dtype, device="cuda")
    b = torch.rand((k, n), dtype=out_dtype, device="cuda")
    bias = torch.rand((1, n), device="cuda", dtype=out_dtype) * 10 + 2.5 if use_bias else None

    #warm up
    for i in range(warmups):
        out = torch.mm(a, b)
        if bias is not None:
            out.add(bias)
    torch.cuda.synchronize()

    start_time = time.time()
    for i in range(iter_num):
        out = torch.mm(a, b)
        if bias is not None:
            out.add(bias)
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time-start_time)/iter_num*1000

    return avg_time

def test_time(name, M,N,K):
    print(f"{name} mm: ({M},{K}) @ ({K},{N})")
    baseline_time = test_torch_mm(M, N, K, output_dtype, True)
    fastdm_fp8_mm_time = test_fastdm_fp8(M, N, K, True, output_dtype)
    fastdm_int8_mm_time = test_fastdm_int8(M, N, K, output_dtype, True, True)

    print(f"fp16 time: {baseline_time}ms")
    print(f"fastdm fp8 time: {fastdm_fp8_mm_time}ms")
    print(f"fastdm int8 time: {fastdm_int8_mm_time}ms")

test_time("qkv_1", 8192, 9216, 3072)
test_time("qkv_2", 512, 9216, 3072)
test_time("ffn_1", 8192, 12288, 3072)
test_time("ffn_2", 8192, 3072, 12288)
test_time("ffn_3", 8704, 3072, 15312)
