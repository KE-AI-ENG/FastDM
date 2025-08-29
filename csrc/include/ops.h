#pragma once

#include <optional>
#include <torch/library.h>

#include <vector>

////////////////elementwise ops begin/////////////////
void int8_quant(torch::Tensor& out,
                            torch::Tensor const& input,
                            torch::Tensor& scales, c10::optional<torch::Tensor> const& azp);
void fp8_quant(
    torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale,
    c10::optional<torch::Tensor> const& scale_ub);
void rms_norm(torch::Tensor& out,
              torch::Tensor& input,
              torch::Tensor& weight,
              double epsilon);

void rotary_embedding(
    torch::Tensor& positions,  // [batch_size, seq_len] or [num_tokens]
    torch::Tensor& query,  // [batch_size, seq_len, num_heads * head_size] or
                           // [num_tokens, num_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    torch::Tensor& key,    // [batch_size, seq_len, num_kv_heads * head_size] or
                           // [num_tokens, num_kv_heads * head_size] or
                           // [batch_size, seq_len, num_heads, head_size] or
                           // [num_tokens, num_heads, head_size]
    int64_t head_size,
    torch::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    bool is_neox);
/////////////////////////////elementwise ops end/////////////////////////////


#if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000) || \
(defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 900 && HOST_CUDA_ARCH < 1000)
std::vector<torch::Tensor> 
flash_mha(torch::Tensor &q,         // batch_size x seqlen_q x num_heads x head_size
        const torch::Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
        const torch::Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
        const float softmax_scale,
        bool is_causal
        );
void fp8_scaled_mm_sm90(torch::Tensor& out, torch::Tensor const& mat_a,
                            torch::Tensor const& mat_b,
                            torch::Tensor const& scales_a,
                            torch::Tensor const& scales_b,
                            const torch::Dtype& out_dtype,
                            c10::optional<torch::Tensor> const& bias);

void int8_scaled_mm_sm90(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     torch::Tensor const& a_scales,
                                     torch::Tensor const& b_scales,
                                     torch::Tensor const& azp_adj,
                                     torch::Tensor const& azp,
                                     std::optional<torch::Tensor> const& bias);
#endif

#if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 890 && __CUDA_ARCH__ < 900) || \
(defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 890 && HOST_CUDA_ARCH < 900)
void fp8_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& mat_a,
                            torch::Tensor const& mat_b,
                            torch::Tensor const& scales_a,
                            torch::Tensor const& scales_b,
                            const torch::Dtype& out_dtype,
                            c10::optional<torch::Tensor> const& bias);

void int8_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                torch::Tensor const& azp,
                                std::optional<torch::Tensor> const& bias);
#endif

#if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 800 && __CUDA_ARCH__ < 890) || \
(defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 800 && HOST_CUDA_ARCH < 890)
void int8_scaled_mm_sm80(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                torch::Tensor const& azp,
                                std::optional<torch::Tensor> const& bias);
#endif

#if (defined __CUDA_ARCH__ && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 800) || \
(defined HOST_CUDA_ARCH && HOST_CUDA_ARCH >= 750 && HOST_CUDA_ARCH < 800)
void int8_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                torch::Tensor const& azp,
                                std::optional<torch::Tensor> const& bias);
#endif
