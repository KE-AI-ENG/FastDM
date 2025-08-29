// Adapted from
// https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::float_e4m3_t, 64>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_fp8<cutlass::float_e4m3_t>(params, stream);
}
