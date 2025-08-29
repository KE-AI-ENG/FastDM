import torch

from fastdm.kernel.registry import kernel_registry

@kernel_registry.register('rmsnorm', 'torch')
def rmsnorm_torch(input: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """ Apply RMS normalization to the input tensor.

    Args:
        input: The input tensor to be normalized.
        scale: The scaling factor for normalization.
        eps: A small value to avoid division by zero.

    Returns:
        torch.Tensor: The normalized tensor.
    """
    input_dtype = input.dtype
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if scale is not None:
        input = input.to(scale.dtype)
        output_tensor = input * scale
    else:
        output_tensor = input.to(input_dtype)

    return output_tensor