import torch
from fastdm.kernel.registry import kernel_registry

@kernel_registry.register('gelu_and_mul', 'torch')
def gelu_and_mul_torch(x):
    """ Apply GELU activation and multiplication to the input tensor.

    Args:
        x: The input tensor to be processed.

    Returns:
        torch.Tensor: The output tensor after applying GELU and multiplication.
    """
    x1, x2 = x.chunk(2, dim=-1)
    out =  x1 * torch.nn.functional.gelu(x2)
    return out