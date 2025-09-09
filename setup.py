import os
import sys

try:
    import torch
except ImportError:
    sys.exit("Error: PyTorch not found.")

def gen_compile_args_from_compute_cap(GPU_Compute_Capability_Major, GPU_Compute_Capability_Minor):

    compile_dicts = {
        "sources": [],
        "extra_compile_args": {},
        "cuda_arch_v": 0
    }

    compile_dicts['cuda_arch_v'] = GPU_Compute_Capability_Major*100+GPU_Compute_Capability_Minor*10

    if 750 == compile_dicts['cuda_arch_v']: #Turing
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu", 'csrc/gemm/turing_w8a8_int8.cu']
        compile_dicts['extra_compile_args'] = {
                                            'nvcc': [
                                                '-O3', 
                                                "-std=c++17",
                                                '--compiler-options', '-fPIC',
                                                '-gencode=arch=compute_75, code=sm_75',
                                            ]
        }
    elif 800 == compile_dicts['cuda_arch_v'] or 860 == compile_dicts['cuda_arch_v'] or 870 == compile_dicts['cuda_arch_v']: #Ampere
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu", 'csrc/gemm/ampere_w8a8_int8.cu']
        compile_dicts['extra_compile_args'] = {
                                            'nvcc': [
                                                '-O3', 
                                                "-std=c++17",
                                                '--compiler-options', '-fPIC',
                                                '-gencode=arch=compute_80, code=sm_80',
                                                '-gencode=arch=compute_86, code=sm_86',
                                                '-gencode=arch=compute_87, code=sm_87'
                                            ]
        }
    elif 890 == compile_dicts['cuda_arch_v']: #Ada Lovelance
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', "csrc/elmwise_ops.cu", 'csrc/gemm/ada_w8a8_fp8.cu', 'csrc/gemm/ada_w8a8_int8.cu']
        compile_dicts['extra_compile_args'] = {
                                            'nvcc': [
                                                '-DNDEBUG',
                                                '-O3', 
                                                "-std=c++17",
                                                '--compiler-options', '-fPIC',
                                                '-gencode=arch=compute_89, code=sm_89',
                                            ]
        }
    elif 900 == compile_dicts['cuda_arch_v']: #Hopper
        compile_dicts["sources"] = ['csrc/torch_bindings.cpp', 'csrc/gemm/hopper_w8a8_fp8.cu', 'csrc/gemm/hopper_w8a8_int8.cu', "csrc/elmwise_ops.cu"] + \
        [os.path.join("csrc/attention", f) for f in os.listdir("./csrc/attention") if f.endswith(".cu")]
        compile_dicts['extra_compile_args'] = {
                                        'nvcc': [
                                                "-DNDEBUG",
                                                "-O3",
                                                "-Xcompiler",
                                                "-fPIC",
                                                "-gencode=arch=compute_90,code=sm_90",
                                                "-gencode=arch=compute_90a,code=sm_90a",
                                                "-std=c++17",
                                                "-DCUTE_USE_PACKED_TUPLE=1",
                                                "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                                                "-DCUTLASS_VERSIONS_GENERATED",
                                                "-DCUTLASS_TEST_LEVEL=0",
                                                "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
                                                "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
                                                "--expt-relaxed-constexpr",
                                                "--expt-extended-lambda",
                                                "--use_fast_math",
                                                "--threads=32",
                                                "-U__CUDA_NO_HALF_OPERATORS__",
                                                "-U__CUDA_NO_HALF_CONVERSIONS__",
                                                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                                                "-U__CUDA_NO_HALF2_OPERATORS__"
                                                #add
                                                # "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                                                # "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                                                # "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                                                # "--ptxas-options=-v",  # printing out number of registers
                                                # '--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage',
                                                # '--ptxas-options=--warn-on-spills',
                                                # '--resource-usage',
                                                # '--source-in-ptx',
                                                # "-lineinfo",
                                        ]
        }
    else:
        sys.exit(f"No implemented for current compute capability: {GPU_Compute_Capability_Major}.{GPU_Compute_Capability_Minor}")

    return compile_dicts


from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if torch.cuda.is_available():
    # Get the compute capability of the current CUDA device
    major, minor = torch.cuda.get_device_capability()
    print(f"=========CUDA Compute Capability: {major}.{minor}==========")
else:
    sys.exit("CUDA is not available.")

#cuda library path
cuda_lib_path = os.environ.get('CUDA_LIB_PATH', '/usr/local/cuda/lib64')
if not os.path.exists(cuda_lib_path):
    sys.exit(f"CUDA library path does not exist: {cuda_lib_path}")

compile_args = gen_compile_args_from_compute_cap(major, minor)

setup(
    name="fastdm",
    version="1.1",
    author="KE-MLSys",
    license="MIT License",
    description=("A lightweight and concise implementation of Diffusion Models Inference"),
    packages=find_packages(exclude=("assets", "csrc", "examples", "tests*", "comfyui", "doc")),
    ext_modules=[
        CUDAExtension(
            name='fastdm.cuda_ops',
            sources=compile_args['sources'],
            define_macros=[('HOST_CUDA_ARCH', compile_args['cuda_arch_v']),],
            extra_compile_args=compile_args['extra_compile_args'],
            include_dirs=[
                os.path.join(os.getcwd(), 'csrc/include'),
                os.path.join(os.getcwd(), 'csrc/include/cutlass/include'),
                os.path.join(os.getcwd(), 'csrc/include/cutlass/tools/util/include'),
                os.path.join(os.getcwd(), 'csrc/include/attention'),
            ],
            libraries=['cuda'],
            library_dirs=[cuda_lib_path],
        )
    ],
    python_requires=">=3.9",
    cmdclass={
        'build_ext': BuildExtension
    }
)