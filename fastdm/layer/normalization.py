# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py

import numbers
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Size
import torch.nn as nn
import torch.nn.functional as F

from fastdm.layer.qlinear import QLinear

from fastdm.kernel.operators_set import rms_norm

class RMSNorm:
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

        self.elementwise_affine = elementwise_affine

    def forward(self, hidden_states):
        if self.weight is None:
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            hidden_states = hidden_states.to(input_dtype)
        else:
            hidden_states = rms_norm(hidden_states, self.weight, self.eps)

        return hidden_states

class SD35AdaLayerNormZeroX:
    r"""
    Norm layer adaptive layer norm zero (AdaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, norm_type: str = "layer_norm", bias: bool = True, eps=1e-5, data_type = torch.bfloat16) -> None:
        super().__init__()

        self.linear = QLinear(embedding_dim, 9 * embedding_dim, bias=bias, data_type=data_type)
        if norm_type == "layer_norm":
            if elementwise_affine:
                self.norm_gamma = torch.ones((embedding_dim), dtype = data_type)
                self.norm_beta = torch.zeros((embedding_dim), dtype = data_type)
            else:
                self.norm_gamma = None
                self.norm_beta = None
            self.norm_shape = (embedding_dim,)
            self.norm_eps = eps
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")
        
        self.elementwise_affine = elementwise_affine

    def forward(
        self,
        hidden_states: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:

        emb = self.linear.forward(F.silu(emb).to(hidden_states.dtype))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
            9, dim=1
        )

        norm_hidden_states = F.layer_norm(hidden_states, self.norm_shape, self.norm_gamma, self.norm_beta, self.norm_eps)

        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2


class AdaLayerNormContinuous:
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        data_type = torch.bfloat16
    ):
        super().__init__()

        self.linear = QLinear(conditioning_embedding_dim, embedding_dim * 2, bias=bias, data_type=data_type)
        if norm_type == "layer_norm":
            if elementwise_affine:
                self.norm_gamma = torch.ones((embedding_dim), dtype = data_type)
                self.norm_beta = torch.zeros((embedding_dim), dtype = data_type)
            else:
                self.norm_gamma = None
                self.norm_beta = None
            self.norm_shape = (embedding_dim,)
            self.norm_eps = eps
        else:
            raise ValueError(f"unknown norm_type {norm_type}")
        
        self.elementwise_affine = elementwise_affine

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear.forward(F.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = F.layer_norm(x, self.norm_shape, self.norm_gamma, self.norm_beta, self.norm_eps) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x

class FP32LayerNorm:
    
    def __init__(
        self,
        normalized_shape: Union[int, list[int], Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = torch.ones(self.normalized_shape, dtype=torch.float32)
            if bias:
                self.bias = torch.zeros(self.normalized_shape, dtype=torch.float32)
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_ = F.layer_norm(input.float(), self.normalized_shape, self.weight, self.bias, self.eps)
        return output_.to(input.dtype)

class AdaLayerNormZero:
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, norm_type="layer_norm", bias=True, data_type=torch.bfloat16):
        super().__init__()

        self.linear = QLinear(embedding_dim, 6 * embedding_dim, bias=bias, data_type=data_type)
        if norm_type == "layer_norm":
            if elementwise_affine:
                self.norm_gamma = torch.ones((embedding_dim), dtype = data_type)
                self.norm_beta = torch.zeros((embedding_dim), dtype = data_type)
            else:
                self.norm_gamma = None
                self.norm_beta = None
            self.norm_shape = (embedding_dim,)
            self.norm_eps = 1e-6
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'."
            )

        self.elementwise_affine = elementwise_affine

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear.forward(F.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = F.layer_norm(x, self.norm_shape, self.norm_gamma, self.norm_beta, self.norm_eps) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp
    
class AdaLayerNormZeroSingle:
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, elementwise_affine=True, norm_type="layer_norm", bias=True, data_type = torch.bfloat16):
        super().__init__()
        if elementwise_affine:
            self.norm_gamma = torch.ones((embedding_dim), dtype = data_type)
            self.norm_beta = torch.zeros((embedding_dim), dtype = data_type)
        else:
            self.norm_gamma = None
            self.norm_beta = None

        self.linear = QLinear(embedding_dim, 3 * embedding_dim, data_type=data_type)

        self.norm_shape = (embedding_dim,)
        self.norm_eps = 1e-6

        self.elementwise_affine = elementwise_affine

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        emb = self.linear.forward(F.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = F.layer_norm(x, self.norm_shape, self.norm_gamma, self.norm_beta, self.norm_eps) * (1 + scale_msa[:, None]) + shift_msa[:, None]

        return x, gate_msa