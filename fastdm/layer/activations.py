# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py

import torch
import torch.nn.functional as F

from fastdm.layer.qlinear import QLinear

class FP32SiLU:
    r"""
    SiLU activation function with input upcasted to torch.float32.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


class GELU:
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, data_type=torch.bfloat16):
        super().__init__()
        self.approximate = approximate

        self.proj = QLinear(dim_in, dim_out, bias=bias, data_type=data_type)

    def forward(self, hidden_states):
        hidden_states = self.proj.forward(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate=self.approximate)
        return hidden_states


class GEGLU:
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, data_type=torch.bfloat16):
        super().__init__()
        self.proj = QLinear(dim_in, dim_out * 2, bias=bias, data_type=data_type)

    def forward(self, hidden_states, *args, **kwargs):
        hidden_states = self.proj.forward(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class SwiGLU:
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function. It's similar to `GEGLU`
    but uses SiLU / Swish instead of GeLU.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, data_type=torch.bfloat16):
        super().__init__()
        self.proj = QLinear(dim_in, dim_out * 2, bias=bias, data_type=data_type)

    def forward(self, hidden_states):
        hidden_states = self.proj.forward(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        return hidden_states * F.silu(gate)


class ApproximateGELU:
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, data_type=torch.bfloat16):
        super().__init__()
        self.proj = QLinear(dim_in, dim_out, bias=bias, data_type=data_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj.forward(x)
        return x * torch.sigmoid(1.702 * x)