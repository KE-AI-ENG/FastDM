
from fastdm.kernel.registry import kernel_registry

@kernel_registry.register('gelu_and_mul', 'cuda')
def gelu_and_mul_cuda(x):
    """
    The gelu-and-mul here is somewhat different from the large language model. Its scope is opposite.
    It apply GELU activation to the second half of the input tensor and multiply it with the first half.
    x = x[:d] * GELU(x[d:]) where d = x.shape[-1] // 2.

    Args:
        input: The input tensor to be processed.

    Returns:
        torch.Tensor: The output tensor after applying GELU and multiplication.
    """
    raise NotImplementedError("GELU and multiplication is not implemented for cuda backend.")

    return out