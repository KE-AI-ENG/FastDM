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
    typename CtaShape,
    typename WarpShape,
    int Stages,
    bool WithBias,
    typename FP8MathOperator = cutlass::arch::OpMultiplyAdd,
    template <typename...> typename EpilogueVisitor = cutlass::epilogue::threadblock::Sm80EVT,
    typename ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>
struct DeviceGemmFp8RowwiseSm89 {
  static_assert(std::is_same_v<ElementType, cutlass::float_e4m3_t>, "ElementType must be FP8(e4m3)");

  using ElementA = ElementType;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementType;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementC = OutElementType;
  using LayoutC = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementOutput = OutElementType;
  using LayoutOutput = cutlass::layout::RowMajor;
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ElementAccumulator = AccumElementType;
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm89;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
  // Number of epilogue stages in EVT
  static constexpr int EVTEpilogueStages = 1;

  using OutputTileThreadMap = cutlass::epilogue::threadblock::
      OutputTileThreadLayout<CtaShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

  // Definition of EVT
  using accSrc = cutlass::epilogue::threadblock::VisitorAccFetch;

  using ComputeBScale = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies,
      ElementComputeEpilogue,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using bScaleSrc = cutlass::epilogue::threadblock::
      VisitorRowBroadcast<OutputTileThreadMap, ElementComputeEpilogue, Stride<_0, _1, _0>>;
  using EpilogueBScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeBScale, accSrc, bScaleSrc>;

  using ComputeAScale = cutlass::epilogue::threadblock::
      VisitorCompute<cutlass::multiplies, ElementC, ElementComputeEpilogue, cutlass::FloatRoundStyle::round_to_nearest>;
  using aScaleSrc = cutlass::epilogue::threadblock::
      VisitorColBroadcast<OutputTileThreadMap, ElementComputeEpilogue, Stride<_1, _0, _0>>;
  using EpilogueAScale = cutlass::epilogue::threadblock::Sm80EVT<ComputeAScale, EpilogueBScale, aScaleSrc>;

  // With bias
  using biasSrc =
      cutlass::epilogue::threadblock::VisitorRowBroadcast<OutputTileThreadMap, ElementOutput, Stride<_0, _1, _0>>;
  using ComputeAScaleWithBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiply_add,
      ElementC,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EpilogueAScaleWithBias =
      cutlass::epilogue::threadblock::Sm80EVT<ComputeAScaleWithBias, EpilogueBScale, aScaleSrc, biasSrc>;

  using dTar = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputTileThreadMap,
      ElementC,
      cutlass::FloatRoundStyle::round_to_nearest,
      Stride<int64_t, _1, _0>>;
  using EpilogueStore = typename cutlass::platform::conditional<
      WithBias,
      cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScaleWithBias>,
      cutlass::epilogue::threadblock::Sm80EVT<dTar, EpilogueAScale>>::type;

  using EpilogueOp = EpilogueStore;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      ElementA,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      AlignmentA,
      ElementB,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      AlignmentB,
      ElementC,
      LayoutC,
      AlignmentC,
      ElementAccumulator,
      ElementComputeEpilogue,
      OperatorClass,
      ArchTag,
      CtaShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      ThreadblockSwizzle,
      Stages,
      FP8MathOperator,
      EVTEpilogueStages>::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Gemm, bool WithBias>
typename Gemm::Arguments prepare_sm89_fp8_args(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ElementT = typename Gemm::ElementA;
  using ElementOutput = typename Gemm::ElementD;
  using ElementComputeEpilogue = float;

  int32_t m = a.size(0);
  int32_t n = b.size(1);
  int32_t k = a.size(1);

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

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

  typename Gemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,  // Mode
      {m, n, k},                                // Problem size
      1,                                        // Split-k factor
      {},                                       // Epilogue args
      ptr_a,                                    // a pointer
      ptr_b,                                    // b pointer
      nullptr,                                  // c pointer (unused)
      nullptr,                                  // d pointer (unused)
      m * k,                                    // batch stride a (unused)
      n * k,                                    // batch stride b (unused)
      m * n,                                    // batch stride c (unused)
      m * n,                                    // batch stride d (unused)
      lda,                                      // stride a
      ldb,                                      // stride b
      ldc,                                      // stride c (unused)
      ldc);                                     // stride d (unused)
  if constexpr (WithBias) {
    args.epilogue = {
        {
            {
                {},  // Accumulator
                {ptr_scales_b, ElementComputeEpilogue(0), {_0{}, _1{}, _0{}}},
                {}  // Multiplies
            },
            {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
            {ptr_bias, ElementOutput(0), {_0{}, _1{}, _0{}}},
            {}  // Multiplies
        },
        {ptr_d, {n, _1{}, _0{}}}};
  } else {
    args.epilogue = {
        {
            {
                {},  // Accumulator
                {ptr_scales_b, ElementComputeEpilogue(0), {_0{}, _1{}, _0{}}},
                {}  // Multiplies
            },
            {ptr_scales_a, ElementComputeEpilogue(0), {_1{}, _0{}, _0{}}},
            {}  // Multiplies
        },
        {ptr_d, {n, _1{}, _0{}}}};
  }

  return args;
}

template <typename Gemm, bool WithBias>
void launch_sm89_fp8_scaled_mm(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  auto args = prepare_sm89_fp8_args<Gemm, WithBias>(out, a, b, scales_a, scales_b, bias);
  Gemm gemm_op;

  size_t workspace_size = gemm_op.get_workspace_size(args);
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto workspace = torch::empty(workspace_size, workspace_options);
  auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

  auto can_implement = gemm_op.can_implement(args);
  TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

  auto status = gemm_op(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess)
}

template <typename OutType, typename CtaShape, typename WarpShape, int Stages>
void sm89_fp8_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = OutType;
  using AccumElementType = float;
  if (bias) {
    using Gemm = typename DeviceGemmFp8RowwiseSm89<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CtaShape,
        WarpShape,
        Stages,
        true>::Gemm;
    return launch_sm89_fp8_scaled_mm<Gemm, true>(out, a, b, scales_a, scales_b, bias);
  } else {
    using Gemm = typename DeviceGemmFp8RowwiseSm89<
        ElementInput,
        ElementOutput,
        AccumElementType,
        CtaShape,
        WarpShape,
        Stages,
        false>::Gemm;
    return launch_sm89_fp8_scaled_mm<Gemm, false>(out, a, b, scales_a, scales_b, bias);
  }
}

template <typename OutType>
void sm89_fp8_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const c10::optional<torch::Tensor>& bias) {
  uint32_t const m = a.size(0);
  uint32_t const n = out.size(1);

  if (m == 1) {
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 16) {
    // M in (1, 16]
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          4>(out, a, b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 64) {
    // M in (16, 64]
    if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<16, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 128) {
    // M in (64, 128]
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          4>(out, a, b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<64, 64, 128>,
          cutlass::gemm::GemmShape<32, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<32, 64, 128>,
          cutlass::gemm::GemmShape<16, 64, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 256) {
    // M in (128, 256]
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 64, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          5>(out, a, b, scales_a, scales_b, bias);
    } else if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<64, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          7>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 64, 128>,
          cutlass::gemm::GemmShape<64, 32, 128>,
          4>(out, a, b, scales_a, scales_b, bias);
    }
  } else if (m <= 512) {
    // M in (256, 512)
    if (n <= 16384) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          2>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          4>(out, a, b, scales_a, scales_b, bias);
    }
  } else {
    // M in (512, inf)
    if (n <= 8192) {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          3>(out, a, b, scales_a, scales_b, bias);
    } else {
      return sm89_fp8_dispatch_bias<
          OutType,
          cutlass::gemm::GemmShape<128, 128, 64>,
          cutlass::gemm::GemmShape<64, 32, 64>,
          2>(out, a, b, scales_a, scales_b, bias);
    }
  }
}

void fp8_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& mat_a,
                            torch::Tensor const& mat_b,
                            torch::Tensor const& scales_a,
                            torch::Tensor const& scales_b,
                            const torch::Dtype& out_dtype,
                            c10::optional<torch::Tensor> const& bias) {

    if (out_dtype == torch::kBFloat16) {
      sm89_fp8_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    } else {
      sm89_fp8_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, scales_a, scales_b, bias);
    }
    return;

}