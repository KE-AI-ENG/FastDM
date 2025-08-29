// Adapted from
// https://github.com/sgl-project/sglang/blob/main/sgl-kernel/csrc/gemm/fp8_gemm_kernel.cu

#include <ATen/cuda/CUDAContext.h>
#include <cudaTypedefs.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

template <
    typename ElementType,
    typename OutElementType,
    typename AccumElementType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename TileSchedulerType = void,
    bool WithBias = false>
struct DeviceGemmFp8RowwiseSm90 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

  // A matrix configuration
  using ElementA = ElementType;               // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = ElementType;                  // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                    // matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementC = void;                      // Element type for C matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C matrix operands
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<OutElementType>::value;  // Memory access granularity/alignment of C matrices in
                                                          // units of elements (up to 16 bytes)

  // Output matrix configuration
  using ElementOutput = OutElementType;            // Element type for output matrix operands
  using LayoutOutput = cutlass::layout::RowMajor;  // Layout type for output matrix operands
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // // Auxiliary matrix configuration and other fusion types
  // using ElementBias = float;

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator = AccumElementType;  // Element type for internal accumulation
  using ElementCompute = float;                 // Element type for compute
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = CTAShape;                            // Threadblock-level tile size

  static constexpr bool PONG = false;
  static constexpr bool FAST_ACCUM = true;
  static constexpr bool USE_BIAS = false;

  using StageCountType = cutlass::gemm::collective::StageCountAuto;      // Stage count maximized
                                                                         // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;  // Kernel to launch based on the default
                                                                         // setting in the Collective Builder
  // Implement rowwise scaling epilogue.
  using XScale = cutlass::epilogue::fusion::Sm90ColBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

  using WScale = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementOutput,
      ElementOutput,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  using Compute0 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementComputeEpilogue,  // First stage output type.
      ElementComputeEpilogue,  // First stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute0 = cutlass::epilogue::fusion::Sm90EVT<Compute0, WScale, Accum>;

  using Compute1 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      ElementOutput,
      ElementComputeEpilogue,  // Second stage input types.
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, XScale, EVTCompute0>;

  // With bias
  using ComputeWithBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiply_add,
      ElementOutput,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<ComputeWithBias, XScale, EVTCompute0, Bias>;

  using EpilogueEVT = typename cutlass::platform::conditional<WithBias, EVTComputeWithBias, EVTCompute1>::type;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementComputeEpilogue,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementOutput,
      LayoutOutput,
      AlignmentOutput,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueEVT>::CollectiveOp;

  using DefaultSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
  using PongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using FastDefaultSchedule = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using FastPongSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;

  using SlowAccum = DefaultSchedule;
  using FastAccum = FastPongSchedule;  // Default apply Pingpong

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm, bool WithBias>
typename Gemm::Arguments prepare_sm90_fp8_args(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ElementT = typename Gemm::ElementA;
  using ElementOutput = typename Gemm::ElementD;
  using ElementComputeEpilogue = float;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);
  ElementT const* ptr_a = reinterpret_cast<ElementT const*>(a.data_ptr());
  ElementT const* ptr_b = reinterpret_cast<ElementT const*>(b.data_ptr());
  ElementOutput const* ptr_bias = nullptr;
  if constexpr (WithBias) {
    TORCH_CHECK(bias.has_value())
    ptr_bias = reinterpret_cast<ElementOutput const*>(bias.value().data_ptr());
  }
  ElementOutput* ptr_d = reinterpret_cast<ElementOutput*>(out.data_ptr());
  ElementComputeEpilogue const* ptr_scales_a = reinterpret_cast<ElementComputeEpilogue const*>(scales_a.data_ptr());
  ElementComputeEpilogue const* ptr_scales_b = reinterpret_cast<ElementComputeEpilogue const*>(scales_b.data_ptr());

  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
  typename Gemm::Arguments args = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {ptr_a, stride_a, ptr_b, stride_b},
      {{},  // epilogue.thread
       nullptr,
       stride_c,
       ptr_d,
       stride_d}};
  if constexpr (WithBias) {
    args.epilogue.thread = {
        {ptr_scales_a},
        {
            {ptr_scales_b},
            {},  // Accumulator
            {}   // Multiplies
        },
        {ptr_bias},
        {},  // Multiplies
    };
  } else {
    args.epilogue.thread = {
        {ptr_scales_a},
        {
            {ptr_scales_b},
            {},  // Accumulator
            {}   // Multiplies
        },
        {},  // Multiplies
    };
  }

  return args;
}

template <typename Gemm, bool WithBias>
void launch_sm90_fp8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  auto args = prepare_sm90_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);
  Gemm gemm_op;

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

  auto status = gemm_op.run(args, workspace.data_ptr(), stream);

  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

template <
    typename OutType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename TileSchedulerType>
void sm90_fp8_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias,
    bool fast_accum = true,
    bool use_persistent = false) {
  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = OutType;
  using AccumElementType = float;
  using EpilogueScheduleType = cutlass::epilogue::TmaWarpSpecialized;

  if (bias) {
    using Gemm = typename DeviceGemmFp8RowwiseSm90<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CTAShape,
        ClusterShape,
        MainloopScheduleType,
        EpilogueScheduleType,
        TileSchedulerType,
        true>::Gemm;
    return launch_sm90_fp8_scaled_mm<Gemm, true>(out, a, b, scales_a, scales_b, bias);
  } else {
    using Gemm = typename DeviceGemmFp8RowwiseSm90<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CTAShape,
        ClusterShape,
        MainloopScheduleType,
        EpilogueScheduleType,
        TileSchedulerType,
        false>::Gemm;
    return launch_sm90_fp8_scaled_mm<Gemm, false>(out, a, b, scales_a, scales_b, bias);
  }
}

template <typename OutType>
void sm90_fp8_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  uint32_t const m = a.size(0);
  using FastPingpongScheduler = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
  using FastBasicScheduler = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
  using PersistentTileScheduler = cutlass::gemm::PersistentScheduler;
  using BasicTileScheduler = void;
  if (m <= 1) {
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_64, _64, _128>,
        Shape<_1, _8, _1>,
        FastBasicScheduler,
        BasicTileScheduler>(out, a, b, scales_a, scales_b, bias);
  }
  if (m <= 64) {
    // m in [1, 64]
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_64, _64, _128>,
        Shape<_1, _4, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  } else if (m <= 256) {
    // m in (64, 256]
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_64, _64, _128>,
        Shape<_1, _1, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  } else if (m <= 1024) {
    // m in (256, 1024]
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_128, _128, _128>,
        Shape<_1, _1, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  } else {
    // m in (1024, inf)
    return sm90_fp8_dispatch_bias<
        OutType,
        Shape<_128, _128, _128>,
        Shape<_2, _1, _1>,
        FastPingpongScheduler,
        PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
  }
}

void fp8_scaled_mm_sm90(torch::Tensor& out,
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias) {

    if (out_dtype == torch::kBFloat16) {
      sm90_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm90_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
    return;

}
