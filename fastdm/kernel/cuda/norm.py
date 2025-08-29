import torch

from fastdm.kernel.registry import kernel_registry

import fastdm.cuda_ops as cuda_ops

@kernel_registry.register('rmsnorm', 'cuda')
def rmsnorm_cuda(input: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """ Apply RMS normalization to the input tensor.

    Args:
        input: The input tensor to be normalized.
        scale: The scaling factor for normalization.
        eps: A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    output_tensor = torch.empty_like(input)
    cuda_ops.rms_norm_(output_tensor, input, scale, eps)
    return output_tensor