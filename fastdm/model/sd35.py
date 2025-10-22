# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_sd3.py

from typing import Any, Dict, List, Optional, Tuple
from collections import namedtuple

import torch
import torch.nn.functional as F
import numpy as np

from fastdm.model.basemodel import BaseModelCore
from fastdm.layer.embeddings import PatchEmbed, CombinedTimestepTextProjEmbeddings
from fastdm.layer.normalization import AdaLayerNormContinuous, AdaLayerNormZero, SD35AdaLayerNormZeroX
from fastdm.layer.transformer import Attention, FeedForward
from fastdm.layer.qlinear import QLinear
from fastdm.caching.xcaching import AutoCache

def _chunked_feed_forward(ff, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output

class JointTransformerBlock:
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = False,
        data_type: torch.dtype = torch.bfloat16, 
    ):
        super().__init__()

        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        if use_dual_attention:
            self.norm1 = SD35AdaLayerNormZeroX(dim, elementwise_affine=False, data_type=data_type)
        else:
            self.norm1 = AdaLayerNormZero(dim, elementwise_affine=False, data_type=data_type)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm", data_type=data_type
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim, elementwise_affine=False, data_type=data_type)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            qk_norm=qk_norm,
            eps=1e-6,
            data_type=data_type,
        )

        if use_dual_attention:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                qk_norm=qk_norm,
                eps=1e-6,
                data_type=data_type,
            )
        else:
            self.attn2 = None

        self.norm2_gamma = None
        self.norm2_beta = None
        self.norm2_shape = (dim,)
        self.norm2_eps = 1e-6
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", data_type=data_type)

        if not context_pre_only:
            self.norm2_context_gamma = None
            self.norm2_context_beta = None
            self.norm2_context_shape = (dim,)
            self.norm2_context_eps = 1e-6
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", data_type=data_type)
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1.forward(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1.forward(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context.forward(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context.forward(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn.forward(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2.forward(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = F.layer_norm(hidden_states, self.norm2_shape, self.norm2_gamma, self.norm2_beta, self.norm2_eps)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff.forward, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff.forward(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = F.layer_norm(encoder_hidden_states, self.norm2_context_shape, self.norm2_context_gamma, self.norm2_context_beta, self.norm2_context_eps)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context.forward, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context.forward(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

class SD3TransformerModelCore(BaseModelCore):
    def __init__(self,
                sample_size: int = 128,
                patch_size: int = 2,
                in_channels: int = 16,
                num_layers: int = 24,
                attention_head_dim: int = 64,
                num_attention_heads: int = 24,
                joint_attention_dim: int = 4096,
                caption_projection_dim: int = 1536,
                pooled_projection_dim: int = 2048,
                out_channels: int = 16,
                pos_embed_max_size: int = 384,
                dual_attention_layers: Tuple[
                    int, ...
                ] = (0,1,2,3,4,5,6,7,8,9,10,11,12),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
                qk_norm: Optional[str] = "rms_norm",
                data_type = torch.bfloat16,
                quant_dtype: torch.dtype = torch.float8_e4m3fn,
                cache: AutoCache = None,
                peft_config: Optional[Dict[str, Any]] = None,  # Peft config for LoRA
                ):
        super().__init__(type="DiT")

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels joint_attention_dim sample_size num_attention_heads attention_head_dim patch_size pooled_projection_dim caption_projection_dim num_layers"
        )
        self.config.in_channels = in_channels
        self.config.joint_attention_dim = joint_attention_dim
        self.config.sample_size = sample_size
        self.config.num_attention_heads = num_attention_heads
        self.config.attention_head_dim = attention_head_dim
        self.config.patch_size = patch_size
        self.config.pooled_projection_dim = pooled_projection_dim
        self.config.caption_projection_dim = caption_projection_dim
        self.config.num_layers = num_layers

        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.quant_dtype = quant_dtype

        # cache
        self.cache = cache
        self.cache_config = cache.config if cache is not None else None
        self.enable_caching = self.cache_config.enable_caching if cache is not None else False


        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = QLinear(self.config.joint_attention_dim, self.config.caption_projection_dim, bias=True)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                    data_type=data_type,
                )
                for i in range(self.config.num_layers)
            ]

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)

        self.proj_out = QLinear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.mixed_precision = False

    def _pre_part_loading(self):
        self.pos_embed.proj_weight = self.init_weight(['pos_embed.proj.weight'])
        self.pos_embed.proj_bias = self.init_weight(['pos_embed.proj.bias'])
        self.pos_embed.pos_embed = self.init_weight(['pos_embed.pos_embed'])
        self.init_weight(['time_text_embed.timestep_embedder.linear_1'], self.time_text_embed.timestep_embedder.linear1)
        self.init_weight(['time_text_embed.timestep_embedder.linear_2'], self.time_text_embed.timestep_embedder.linear2)
        self.init_weight(['time_text_embed.text_embedder.linear_1'], self.time_text_embed.text_embedder.linear1)
        self.init_weight(['time_text_embed.text_embedder.linear_2'], self.time_text_embed.text_embedder.linear2)
        self.init_weight(['context_embedder'], self.context_embedder)
        return

    def __transformer_block_loading(self):
        for i in range(len(self.transformer_blocks)):
            #norm
            self.init_weight([f"transformer_blocks.{i}.norm1.linear"], self.transformer_blocks[i].norm1.linear)
            self.init_weight([f"transformer_blocks.{i}.norm1_context.linear"], self.transformer_blocks[i].norm1_context.linear)

            #attention
            self.transformer_blocks[i].attn.norm_q_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_q.weight"])
            self.transformer_blocks[i].attn.norm_k_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_k.weight"])
            self.init_weight([f"transformer_blocks.{i}.attn.to_q", f"transformer_blocks.{i}.attn.to_k", f"transformer_blocks.{i}.attn.to_v"], self.transformer_blocks[i].attn.qkv, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.attn.add_q_proj", f"transformer_blocks.{i}.attn.add_k_proj", f"transformer_blocks.{i}.attn.add_v_proj"], self.transformer_blocks[i].attn.add_qkv_proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.attn.to_out.0"], self.transformer_blocks[i].attn.to_out, self.quant_dtype)
            if self.transformer_blocks[i].attn.context_pre_only is not None and not self.transformer_blocks[i].attn.context_pre_only:
                self.init_weight([f"transformer_blocks.{i}.attn.to_add_out"], self.transformer_blocks[i].attn.to_add_out, self.quant_dtype)
            self.transformer_blocks[i].attn.norm_added_q_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_added_q.weight"])
            self.transformer_blocks[i].attn.norm_added_k_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_added_k.weight"])
            if self.transformer_blocks[i].use_dual_attention:
                self.transformer_blocks[i].attn2.norm_q_weight = self.init_weight([f"transformer_blocks.{i}.attn2.norm_q.weight"])
                self.transformer_blocks[i].attn2.norm_k_weight = self.init_weight([f"transformer_blocks.{i}.attn2.norm_k.weight"])
                self.init_weight([f"transformer_blocks.{i}.attn2.to_q", f"transformer_blocks.{i}.attn2.to_k", f"transformer_blocks.{i}.attn2.to_v"], self.transformer_blocks[i].attn2.qkv, self.quant_dtype)
                self.init_weight([f"transformer_blocks.{i}.attn2.to_out.0"], self.transformer_blocks[i].attn2.to_out, self.quant_dtype)
            
            #feedforward
            self.init_weight([f"transformer_blocks.{i}.ff.net.0.proj"], self.transformer_blocks[i].ff.act_fn.proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.ff.net.2"], self.transformer_blocks[i].ff.ff_out_proj, self.quant_dtype)
            if not self.transformer_blocks[i].context_pre_only:
                self.init_weight([f"transformer_blocks.{i}.ff_context.net.0.proj"], self.transformer_blocks[i].ff_context.act_fn.proj, self.quant_dtype)
                self.init_weight([f"transformer_blocks.{i}.ff_context.net.2"], self.transformer_blocks[i].ff_context.ff_out_proj, self.quant_dtype)
        
        return
    
    def _post_part_loading(self):
        self.init_weight(['norm_out.linear'], self.norm_out.linear, self.quant_dtype)
        self.init_weight(['proj_out'], self.proj_out, self.quant_dtype)
        return
    
    def _major_parts_loading(self):
        self.__transformer_block_loading()

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`SD3TransformerModelCore`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed.forward(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed.forward(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder.forward(encoder_hidden_states)

        if self.enable_caching:
            hidden_states = self.cache.apply_cache(
                model_type="sd35",
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                attention_kwargs=joint_attention_kwargs,
                transformer_blocks=self.transformer_blocks,
            )
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                    interval_control = len(self.transformer_blocks) // len(block_controlnet_hidden_states)
                    hidden_states = hidden_states + block_controlnet_hidden_states[index_block // interval_control]

        hidden_states = self.norm_out.forward(hidden_states, temb)
        hidden_states = self.proj_out.forward(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        return (output,)