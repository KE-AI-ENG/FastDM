#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/numeric_types.h>

#include "ops.h"

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor,
                         0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor,
                         0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}

torch::Tensor fp8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");

  TORCH_CHECK(
      (mat_a.size(1) * mat_a.element_size()) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(
      (mat_b.size(0) * mat_b.element_size()) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(mat_a.scalar_type() == torch::kFloat8_e4m3fn, "mat_a must be Float8_e4m3fn");
  TORCH_CHECK(mat_b.scalar_type() == torch::kFloat8_e4m3fn, "mat_b must be Float8_e4m3fn");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  if (bias) {
    TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match output dtype");
  }

  torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));
  TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  int32_t version_num = get_sm_version_num();

  if (version_num == 89) {
#if defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 890 && HOST_CUDA_ARCH < 900
    fp8_scaled_mm_sm89(out, mat_a, mat_b, scales_a, scales_b, out_dtype, bias);
#endif
    return out;
  }
  else if (version_num == 90) {
#if defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 900 && HOST_CUDA_ARCH < 1000
    fp8_scaled_mm_sm90(out, mat_a, mat_b, scales_a, scales_b, out_dtype, bias);
#endif
    return out;
  }
  else{
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "No implemented fp8_scaled_mm for current compute capability: "
        "CUDA device capability: ",
        version_num);
  }
}

torch::Tensor int8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    torch::Tensor const& azp_adj,
    torch::Tensor const& azp,
    std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "mat_a must be a row major tensor");
  TORCH_CHECK(mat_b.stride(0) == 1, "mat_a must be a column major tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK(mat_a.size(1) % 16 == 0, "mat_a.size(1) must be multiple of 16 for memory alignment");
  TORCH_CHECK(mat_b.size(0) % 16 == 0, "mat_b.size(0) must be multiple of 16 for memory alignment");
  TORCH_CHECK(mat_b.size(1) % 8 == 0, "mat_b.size(1) must be multiple of 8 for memory alignment");  // out.stride(0)
  TORCH_CHECK(mat_a.scalar_type() == torch::kInt8, "mat_a must be Int8");
  TORCH_CHECK(mat_b.scalar_type() == torch::kInt8, "mat_b must be Int8");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  TORCH_CHECK(scales_a.numel() == mat_a.size(0), "size of scales_a is not matched");
  TORCH_CHECK(scales_b.numel() == mat_b.size(1), "size of scales_b is not matched");
  TORCH_CHECK(scales_a.is_contiguous(), "scales_a must be contiguous");
  TORCH_CHECK(scales_b.is_contiguous(), "scales_b msut be contiguous");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "scales_b must be Float32");

  if (bias) {
    TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match output dtype");
  }

  TORCH_CHECK(azp.numel() == mat_a.size(0) && azp.is_contiguous());
  TORCH_CHECK(azp_adj.numel() == mat_b.size(1) && azp_adj.is_contiguous());

  torch::Tensor out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));
  int32_t version_num = get_sm_version_num();

  if (90 == version_num) {
#if defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 900 && HOST_CUDA_ARCH < 1000
    int8_scaled_mm_sm90(out, mat_a, mat_b, scales_a, scales_b, azp_adj, azp, bias);
#endif
    return out;
  }
  else if (89 == version_num) {
#if defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 890 && HOST_CUDA_ARCH < 900
    int8_scaled_mm_sm89(out, mat_a, mat_b, scales_a, scales_b, azp_adj, azp, bias);
#endif
    return out;
  }
  else if (80 == version_num) {
#if defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 800 && HOST_CUDA_ARCH < 890
    int8_scaled_mm_sm80(out, mat_a, mat_b, scales_a, scales_b, azp_adj, azp, bias);
#endif
    return out;
  }
  else if (75 == version_num) {
#if defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 750 && HOST_CUDA_ARCH < 800
    int8_scaled_mm_sm75(out, mat_a, mat_b, scales_a, scales_b, azp_adj, azp, bias);
#endif
    return out;
  }
  else {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "No implemented int8_scaled_mm for current compute capability: "
        "CUDA device capability: ",
        version_num);
  }

}

torch::Tensor flash_attention_fp8_fwd(
        torch::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const torch::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const torch::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        const float softmax_scale,
        bool is_causal) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

  torch::Tensor out;
  int32_t version_num = get_sm_version_num();

  if (90 == version_num) {
#if defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 900 && HOST_CUDA_ARCH < 1000
    out = flash_mha(q, k, v, softmax_scale, is_causal)[0];
#endif
    return out;
  }
  else {
    TORCH_CHECK_NOT_IMPLEMENTED(
        false,
        "No implemented flash-attention for current compute capability: "
        "CUDA device capability: ",
        version_num);
  }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_quant_", &fp8_quant, "dynamic per token scaled fp8 quant");
    m.def("int8_quant_", &int8_quant, "dynamic per token scaled int8 quant");
    m.def("rms_norm_", &rms_norm, "rms norm");
    m.def("rotary_emb_", &rotary_embedding, "rotary embedding");

    m.def("fp8_scaled_mm_", &fp8_scaled_mm, "CUTLASS fp8 Symmetric Scaled Matrix Multiplication");
    m.def("int8_scaled_mm_", &int8_scaled_mm, "CUTLASS int8 ASymmetric Scaled Matrix Multiplication");

    m.def("flash_attention_fp8_fwd_", &flash_attention_fp8_fwd, "Flash Attention fp8 forward");
}