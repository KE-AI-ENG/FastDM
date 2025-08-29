import torch
import triton
import triton.language as tl
from fastdm.kernel.registry import kernel_registry

# Adapted from https://github.com/sgl-project/sglang/blob/v0.5.0rc2/python/sglang/srt/layers/elementwise.py#L399
@triton.jit
def gelu_and_mul_kernel(
    out_hidden_states_ptr,      # (bs, hidden_dim)
    hidden_states_ptr,          # (bs, hidden_dim * 2)
    hidden_dim: tl.constexpr,   # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0).to(tl.float32)

    # gelu
    gelu_x3 = 0.5 * x3 * (1.0 + tl.erf(x3 / tl.sqrt(2.0)))
    out = x1 * gelu_x3.to(hidden_states_ptr.dtype.element_ty)

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


@kernel_registry.register('gelu_and_mul', 'triton')
def gelu_and_mul_triton(x):
    """ Apply GELU activation and multiplication to the input tensor.

    Args:
        x: The input tensor to be processed.

    Returns:
        torch.Tensor: The output tensor after applying GELU and multiplication.
    """
    assert x.ndim == 2
    assert x.is_contiguous() == True
    bs = x.shape[0]
    in_hidden_dim = x.shape[1]
    hidden_dim = in_hidden_dim // 2

    out = torch.empty((bs, hidden_dim), dtype=x.dtype, device=x.device)

    max_warps = 32
    config = {
        # 8 ele per thread (not tuned)
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),
    }

    gelu_and_mul_kernel[(bs,)](
        out,
        x,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        **config,
    )

    return out