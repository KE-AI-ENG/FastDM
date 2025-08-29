import os
from typing import Callable

import torch

def get_available_backends(op_name: str) -> list:
    """Get the list of available backends for the operator"""
    from fastdm.kernel.registry import kernel_registry
    return list(kernel_registry._registry.get(op_name, {}).keys())

def set_global_backend(backend: str):
    """Global Settings Backend"""
    if backend not in ["torch", "triton", "cuda"]:
        raise ValueError(f"Unsupported backend: {backend}. Available backends: torch, triton, cuda.")
    os.environ['KERNEL_BACKEND'] = backend

def benchmark_kernel(op_name: str, impl_func: Callable, iterations: int = 50, *args, **kwargs):
    """Kernel Performance Benchmark"""
    import time
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # warm up
    for _ in range(3):
        impl_func(*args, **kwargs)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        result = impl_func(*args, **kwargs)
    torch.cuda.synchronize()
    duration = (time.perf_counter() - start) / iterations
    
    return duration, result

def kernel_output_assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = None,
    atol: float = None,
):
    torch.testing.assert_close(actual,
                               expected.to(actual.dtype),
                               rtol=rtol,
                               atol=atol)