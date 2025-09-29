# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py

from typing import Any, Dict, Tuple, Optional, List
import numpy as np

import torch
import torch.nn.functional as F

from fastdm.model.basemodel import BaseModelCore
from fastdm.layer.embeddings import FluxPosEmbed, CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from fastdm.layer.normalization import AdaLayerNormZero, AdaLayerNormContinuous, AdaLayerNormZeroSingle
from fastdm.layer.transformer import Attention, FeedForward
from fastdm.layer.qlinear import QLinear
from fastdm.caching.xcaching import AutoCache

class FluxSingleTransformerBlock:
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

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, data_type=torch.bfloat16):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim, elementwise_affine=False, data_type = data_type)
        self.proj_mlp = QLinear(dim, self.mlp_hidden_dim, bias=True, data_type=data_type)
        self.proj_out = QLinear(dim + self.mlp_hidden_dim, dim, bias=True, data_type=data_type)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
            data_type=data_type,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm.forward(hidden_states, emb=temb)
        mlp_hidden_states = F.gelu(self.proj_mlp.forward(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn.forward(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out.forward(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states

class FluxTransformerBlock:
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

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6, data_type=torch.bfloat16):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, elementwise_affine= False, data_type=data_type)

        self.norm1_context = AdaLayerNormZero(dim, elementwise_affine= False, data_type=data_type)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            qk_norm=qk_norm,
            eps=eps,
            data_type=data_type,
        )

        self.norm2_gamma = None
        self.norm2_beta = None
        self.norm2_shape = (dim,)
        self.norm2_eps = 1e-6

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", data_type=data_type)

        self.norm2_context_gamma = None
        self.norm2_context_beta = None
        self.norm2_context_shape = (dim,)
        self.norm2_context_eps = 1e-6
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", data_type=data_type)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1.forward(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context.forward(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        # Attention.
        attn_output, context_attn_output = self.attn.forward(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = F.layer_norm(hidden_states, self.norm2_shape, self.norm2_gamma, self.norm2_beta, self.norm2_eps)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff.forward(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = F.layer_norm(encoder_hidden_states, self.norm2_context_shape, self.norm2_context_gamma, self.norm2_context_beta, self.norm2_context_eps)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context.forward(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

class FluxTransformer2DModelCore(BaseModelCore):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
        axes_dims_rope (`Tuple[int]`, *optional*, defaults to (16, 56, 56)): Axes dimensions for the rotary positional embeddings.
        data_type (`torch.dtype`, *optional*, defaults to `torch.bfloat16`): The data type to use for the model weights.
        quant_dtype (`torch.dtype`, *optional*, defaults to `torch.float8_e4m3fn`): The quantization data type to use for the model matrix multiplications.
        enable_caching (`bool`, *optional*, defaults to False): Whether to enable the TEACache mechanism for caching intermediate results during inference.
        teacache_thresh (`float`, *optional*, defaults to 0.25): The threshold for the relative L1 distance to trigger TEACache updates.
        num_steps (`int`, *optional*, defaults to 25): The number of steps for TEACache count updates.
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        data_type = torch.bfloat16,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
        cache: AutoCache = None,
        oom_ressolve: bool=False, #The pipeline will running in cpu if it set to True, so we need copy tensor to gpu in forward.
    ):
        super().__init__(type="DiT")

        self.need_resolve_oom = oom_ressolve

        self.out_channels = out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim, data_type=data_type
        )

        self.context_embedder = QLinear(joint_attention_dim, self.inner_dim, data_type=data_type)
        self.x_embedder = QLinear(in_channels, self.inner_dim, data_type=data_type)

        self.quant_dtype = quant_dtype

        self.transformer_blocks = [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    data_type=data_type,
                )
                for i in range(num_layers)
            ]

        self.single_transformer_blocks = [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    data_type=data_type,
                )
                for i in range(num_single_layers)
            ]

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6, data_type=data_type)

        self.proj_out = QLinear(self.inner_dim, patch_size * patch_size * self.out_channels, data_type=data_type)

        self.mixed_precision = False

        # cache
        self.cache = cache
        self.cache_config = cache.config if cache is not None else None
        self.enable_caching = self.cache_config.enable_caching if cache is not None else False

    def _pre_part_loading(self):
        self.init_weight(["time_text_embed.timestep_embedder.linear_1"], self.time_text_embed.timestep_embedder.linear1)
        self.init_weight(["time_text_embed.timestep_embedder.linear_2"], self.time_text_embed.timestep_embedder.linear2)
        self.init_weight(["time_text_embed.guidance_embedder.linear_1"], self.time_text_embed.guidance_embedder.linear1)
        self.init_weight(["time_text_embed.guidance_embedder.linear_2"], self.time_text_embed.guidance_embedder.linear2)
        self.init_weight(["time_text_embed.text_embedder.linear_1"], self.time_text_embed.text_embedder.linear1)
        self.init_weight(["time_text_embed.text_embedder.linear_2"], self.time_text_embed.text_embedder.linear2)
        self.init_weight(["context_embedder"], self.context_embedder)
        self.init_weight(["x_embedder"], self.x_embedder)
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
            self.init_weight([f"transformer_blocks.{i}.attn.to_add_out"], self.transformer_blocks[i].attn.to_add_out, self.quant_dtype)
            self.transformer_blocks[i].attn.norm_added_q_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_added_q.weight"])
            self.transformer_blocks[i].attn.norm_added_k_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_added_k.weight"])
            
            #feedforward
            self.init_weight([f"transformer_blocks.{i}.ff.net.0.proj"], self.transformer_blocks[i].ff.act_fn.proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.ff.net.2"], self.transformer_blocks[i].ff.ff_out_proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.ff_context.net.0.proj"], self.transformer_blocks[i].ff_context.act_fn.proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.ff_context.net.2"], self.transformer_blocks[i].ff_context.ff_out_proj, self.quant_dtype)
        
        return
    
    def __single_trans_block_loading(self):
        for i in range(len(self.single_transformer_blocks)):
            #norm
            self.init_weight([f"single_transformer_blocks.{i}.norm.linear"], self.single_transformer_blocks[i].norm.linear)

            #proj
            self.init_weight([f"single_transformer_blocks.{i}.proj_mlp"], self.single_transformer_blocks[i].proj_mlp, self.quant_dtype)
            self.init_weight([f"single_transformer_blocks.{i}.proj_out"], self.single_transformer_blocks[i].proj_out, self.quant_dtype)

            #attn
            self.single_transformer_blocks[i].attn.norm_q_weight = self.init_weight([f"single_transformer_blocks.{i}.attn.norm_q.weight"])
            self.single_transformer_blocks[i].attn.norm_k_weight = self.init_weight([f"single_transformer_blocks.{i}.attn.norm_k.weight"])
            self.init_weight([f"single_transformer_blocks.{i}.attn.to_q", f"single_transformer_blocks.{i}.attn.to_k", f"single_transformer_blocks.{i}.attn.to_v"], self.single_transformer_blocks[i].attn.qkv, self.quant_dtype)

        return
    
    def _post_part_loading(self):
        self.init_weight(["norm_out.linear"], self.norm_out.linear)
        self.init_weight(["proj_out"], self.proj_out)
        return

    def _major_parts_loading(self):
        self.__transformer_block_loading()
        self.__single_trans_block_loading()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        """
        The [`FluxTransformer2DModel`] forward method.

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

        if self.need_resolve_oom: #need copy tensor to gpu
            hidden_states = hidden_states.to(self.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(self.device)
            if pooled_projections is not None:
                pooled_projections = pooled_projections.to(self.device)
            if timestep is not None:
                timestep = timestep.to(self.device)
            if img_ids is not None:
                img_ids = img_ids.to(self.device)
            if txt_ids is not None:
                txt_ids = txt_ids.to(self.device)
            if guidance is not None:
                guidance = guidance.to(self.device)

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        hidden_states = self.x_embedder.forward(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed.forward(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed.forward(timestep, guidance, pooled_projections)
        )

        encoder_hidden_states = self.context_embedder.forward(encoder_hidden_states)

        if txt_ids.ndim == 3:
            # logger.warning(
            #     "Passing `txt_ids` 3d torch.Tensor is deprecated."
            #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
            # )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            # logger.warning(
            #     "Passing `img_ids` 3d torch.Tensor is deprecated."
            #     "Please remove the batch dimension and pass it as a 2d torch Tensor"
            # )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed.forward(ids)
        #merge cos and sin to single tensor, to ultilize the custom-rope-emb-ops
        image_rotary_emb = torch.cat((image_rotary_emb[0][:,0::2], image_rotary_emb[1][:,1::2]), dim=-1).to(hidden_states.dtype)
        
        if self.enable_caching:
            hidden_states = self.cache.apply_cache(
                model_type="flux",
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=joint_attention_kwargs,
                transformer_blocks=self.transformer_blocks,
                single_transformer_blocks=self.single_transformer_blocks,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=controlnet_blocks_repeat
            )
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if controlnet_block_samples is not None:
                    interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    # For Xlabs ControlNet.
                    if controlnet_blocks_repeat:
                        hidden_states = (
                            hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                        )
                    else:
                        hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.single_transformer_blocks):
                hidden_states = block.forward(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if controlnet_single_block_samples is not None:
                    interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                    interval_control = int(np.ceil(interval_control))
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                        + controlnet_single_block_samples[index_block // interval_control]
                    )

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out.forward(hidden_states, temb)

        output = self.proj_out.forward(hidden_states)

        if self.need_resolve_oom:
            output = output.to("cpu")
            
        return (output,)
