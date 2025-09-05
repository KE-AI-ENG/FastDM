# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py

from typing import Any, Dict, Tuple, Optional, Union
import math
import numpy as np

import torch
import torch.nn.functional as F

from fastdm.model.basemodel import BaseModelCore
from fastdm.layer.embeddings import WanRotaryPosEmbed, WanTimeTextImageEmbedding
from fastdm.layer.normalization import FP32LayerNorm
from fastdm.layer.transformer import FeedForward
from fastdm.layer.qlinear import QLinear
from fastdm.layer.transformer import WanAttention
from fastdm.caching.xcaching import AutoCache

class WanTransformerBlock:
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        data_type = torch.bfloat16
    ):
        
        self.cross_attn_norm = cross_attn_norm

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            data_type=data_type
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            data_type=data_type
        )

        if self.cross_attn_norm:
            self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True)

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate", data_type=data_type)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        # self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.scale_shift_table = torch.randn(1, 6, dim) / dim**0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1.forward(hidden_states) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1.forward(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        if self.cross_attn_norm:
            norm_hidden_states = self.norm2.forward(hidden_states).type_as(hidden_states)
        else:
            norm_hidden_states = hidden_states
        attn_output = self.attn2.forward(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3.forward(hidden_states) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn.forward(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

class WanTransformer3DModelCore(BaseModelCore):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
        data_type = torch.bfloat16,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
        cache: AutoCache = None,
    ) -> None:
        super().__init__("DiT")

        self.quant_dtype = quant_dtype

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_kernel_size = patch_size
        self.patch_embedding_stride = patch_size
        self.patch_embedding_weight = torch.empty((inner_dim,in_channels,patch_size[0],patch_size[1],patch_size[2]), dtype = data_type)
        self.patch_embedding_bias = torch.empty((inner_dim), dtype = data_type)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
            data_type=data_type
        )

        # 3. Transformer blocks
        self.blocks = [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim, data_type=data_type
                )
                for _ in range(num_layers)
            ]

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = QLinear(inner_dim, out_channels * math.prod(patch_size), data_type=data_type)
        # self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.scale_shift_table = torch.randn(1, 2, inner_dim) / inner_dim**0.5

        # cache
        self.cache = cache
        self.cache_config = cache.config if cache is not None else None
        self.enable_caching = self.cache_config.enable_caching if cache is not None else False


    def _pre_part_loading(self):
        self.patch_embedding_weight = self.init_weight(['patch_embedding.weight'])
        self.patch_embedding_bias =self.init_weight(['patch_embedding.bias'])

        #condition_embedder
        self.init_weight(["condition_embedder.time_embedder.linear_1"], self.condition_embedder.time_embedder.linear1)
        self.init_weight(["condition_embedder.time_embedder.linear_2"], self.condition_embedder.time_embedder.linear2)
        self.init_weight(["condition_embedder.time_proj"], self.condition_embedder.time_proj)
        self.init_weight(["condition_embedder.text_embedder.linear_1"], self.condition_embedder.text_embedder.linear1)
        self.init_weight(["condition_embedder.text_embedder.linear_2"], self.condition_embedder.text_embedder.linear2)
        if self.condition_embedder.image_embedder is not None:
            self.condition_embedder.image_embedder.norm1.weight = self.init_weight(["condition_embedder.image_embedder.norm1.weight"]).to(torch.float32)
            self.condition_embedder.image_embedder.norm1.bias = self.init_weight(["condition_embedder.image_embedder.norm1.bias"]).to(torch.float32)
            self.init_weight(['condition_embedder.image_embedder.ff.net.0.proj'], self.condition_embedder.image_embedder.ff.act_fn.proj)
            self.init_weight(['condition_embedder.image_embedder.ff.net.2'], self.condition_embedder.image_embedder.ff.ff_out_proj)
            self.condition_embedder.image_embedder.norm2.weight = self.init_weight(["condition_embedder.image_embedder.norm2.weight"]).to(torch.float32)
            self.condition_embedder.image_embedder.norm2.bias = self.init_weight(["condition_embedder.image_embedder.norm2.bias"]).to(torch.float32)
            if self.condition_embedder.image_embedder.pos_embed is not None:
                self.condition_embedder.image_embedder.pos_embed = self.init(['condition_embedder.image_embedder.pos_embed'])
        return
    
    def _post_part_loading(self):
        self.init_weight(['proj_out'], self.proj_out)
        self.scale_shift_table = self.init_weight(['scale_shift_table'])
        return

    def _major_parts_loading(self):
        for i in range(len(self.blocks)):
            #attn1
            self.blocks[i].attn1.norm_q_weight = self.init_weight([f"blocks.{i}.attn1.norm_q.weight"])
            self.blocks[i].attn1.norm_k_weight = self.init_weight([f"blocks.{i}.attn1.norm_k.weight"])
            self.init_weight([f"blocks.{i}.attn1.to_q", f"blocks.{i}.attn1.to_k", f"blocks.{i}.attn1.to_v"], self.blocks[i].attn1.qkv, self.quant_dtype)
            self.init_weight([f"blocks.{i}.attn1.to_out.0"], self.blocks[i].attn1.to_out, self.quant_dtype)

            #attn2
            self.blocks[i].attn2.norm_q_weight = self.init_weight([f"blocks.{i}.attn2.norm_q.weight"])
            self.blocks[i].attn2.norm_k_weight = self.init_weight([f"blocks.{i}.attn2.norm_k.weight"])
            # self.init_weight([f"blocks.{i}.attn2.to_q", f"blocks.{i}.attn2.to_k", f"blocks.{i}.attn2.to_v"], self.blocks[i].attn2.qkv, self.quant_dtype)
            self.init_weight([f"blocks.{i}.attn2.to_q"], self.blocks[i].attn2.to_q, self.quant_dtype)
            self.init_weight([f"blocks.{i}.attn2.to_k", f"blocks.{i}.attn2.to_v"], self.blocks[i].attn2.to_kv, self.quant_dtype)
            self.init_weight([f"blocks.{i}.attn2.to_out.0"], self.blocks[i].attn2.to_out, self.quant_dtype)
            if self.blocks[i].attn2.added_kv_proj_dim is not None:
                self.init_weight([f"blocks.{i}.attn2.add_k_proj"], self.blocks[i].attn2.add_k_proj, self.quant_dtype)
                self.init_weight([f"blocks.{i}.attn2.add_v_proj"], self.blocks[i].attn2.add_v_proj, self.quant_dtype)
                self.blocks[i].attn2.norm_added_k_weight = self.init_weight([f"blocks.{i}.attn2.norm_added_k.weight"])
            
            #norm2
            if self.blocks[i].cross_attn_norm:
                self.blocks[i].norm2.weight = self.init_weight([f"blocks.{i}.norm2.weight"]).to(torch.float32)
                self.blocks[i].norm2.bias = self.init_weight([f"blocks.{i}.norm2.bias"]).to(torch.float32)

            #feedforward
            self.init_weight([f"blocks.{i}.ffn.net.0.proj"], self.blocks[i].ffn.act_fn.proj, self.quant_dtype)
            self.init_weight([f"blocks.{i}.ffn.net.2"], self.blocks[i].ffn.ff_out_proj, self.quant_dtype)

            #scale shift table
            self.blocks[i].scale_shift_table = self.init_weight([f'blocks.{i}.scale_shift_table'])
        return
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            print(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_embedding_kernel_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope.forward(hidden_states)

        # hidden_states = self.patch_embedding.forward(hidden_states)
        hidden_states = F.conv3d(hidden_states, self.patch_embedding_weight, self.patch_embedding_bias, self.patch_embedding_stride)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder.forward(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        if self.enable_caching:
            hidden_states = self.cache.apply_cache(
                model_type="wan",
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                image_rotary_emb=rotary_emb,
                attention_kwargs=attention_kwargs,
                transformer_blocks=self.blocks,
            )
        else:
            for block in self.blocks:
                hidden_states = block.forward(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out.forward(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out.forward(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return (output,)