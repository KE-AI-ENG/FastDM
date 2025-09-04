# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_qwenimage.py

from typing import Any, Dict, Tuple, Optional, List
import numpy as np

import torch
import torch.nn.functional as F

from fastdm.model.basemodel import BaseModelCore
from fastdm.layer.embeddings import QwenEmbedRope, QwenTimestepProjEmbeddings
from fastdm.layer.normalization import RMSNorm, AdaLayerNormContinuous
from fastdm.layer.transformer import Attention, FeedForward
from fastdm.layer.qlinear import QLinear
from fastdm.cache_config import CacheConfig

class QwenImageTransformerBlock:
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6, data_type = torch.bfloat16
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        self.eps = eps

        # Image processing modules
        self.img_mod_proj = QLinear(dim, 6*dim, data_type=data_type)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            qk_norm=qk_norm,
            eps=eps,
            data_type=data_type
        )

        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", data_type=data_type)

        # Text processing modules
        self.txt_mod_proj = QLinear(dim, 6*dim, data_type=data_type)

        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", data_type=data_type)

    def _modulate(self, x, mod_params):
        """Apply modulation to input tensor"""
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        # image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        img_mod_params = self.img_mod_proj.forward(F.silu(temb))
        txt_mod_params = self.txt_mod_proj.forward(F.silu(temb))

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = F.layer_norm(hidden_states, (self.dim,), eps=self.eps)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = F.layer_norm(encoder_hidden_states, (self.dim,), eps=self.eps)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # Use QwenAttnProcessor2_0 for joint attention computation
        # This directly implements the DoubleStreamLayerMegatron logic:
        # 1. Computes QKV for both streams
        # 2. Applies QK normalization and RoPE
        # 3. Concatenates and runs joint attention
        # 4. Splits results back to separate streams
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn.forward_qwen(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = F.layer_norm(hidden_states, (self.dim,), eps=self.eps)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp.forward(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = F.layer_norm(encoder_hidden_states, (self.dim,), eps=self.eps)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp.forward(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states

class QwenImageTransformer2DModelCore(BaseModelCore):
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
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 64,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        data_type = torch.bfloat16,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
        cache_config: CacheConfig = None,
        oom_ressolve: bool=False, #The pipeline will running in cpu if it set to True, so we need copy tensor to gpu in forward.
    ):
        super().__init__("DiT")

        self.quant_dtype = quant_dtype

        self.need_resolve_oom = oom_ressolve

        #This part occupies 12G vram. If quantization is performed, it will affect the generation effect, so we quantize it in <24GB vram cards
        self.quant_img_txt_mod = self.need_resolve_oom and (torch.cuda.get_device_properties().total_memory/(1<<30))<24

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim, data_type=data_type)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self.img_in = QLinear(in_channels, self.inner_dim, data_type=data_type)
        self.txt_in = QLinear(joint_attention_dim, self.inner_dim, data_type=data_type)

        self.transformer_blocks = [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    data_type=data_type
                )
                for _ in range(num_layers)
            ]

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6, data_type=data_type)

        self.proj_out = QLinear(self.inner_dim, patch_size * patch_size * self.out_channels, data_type=data_type)

        # teacahe
        self.cache_config = cache_config
        self.enable_caching = cache_config.enable_caching if cache_config is not None else False
        # Note: # qwenimage will excute the forward separately in one step for prompt and negative prompt.
        if self.enable_caching:
            self.accumulated_rel_l1_distance_dict = {
                "positive": 0,
                "negative": 0
            }
            self.previous_modulated_input_dict = {
                "positive": None,
                "negative": None
            }
            self.previous_residual_dict = {
                "positive": None,
                "negative": None
            }
            self.previous_encoder_residual_dict = {
                "positive": None,
                "negative": None
            }
            self.coefficients = {
                "positive": self.cache_config.coefficients,
                "negative": self.cache_config.negtive_coefficients
            } #txt_modulated
            # self.coefficients = {
            #     "positive": [ 12.67449879, -24.4436427,   14.21152966,  -1.58014773,   0.12182928],
            #     "negative": [ 11.91250519, -22.92346214,  13.13159225,  -1.2740735,    0.10579601]
            # } #tmeb get bad result for comfyui forward.
            self.cache_status = {
                "positive": True,
                "negative": False
            }

    def _pre_part_loading(self):
        self.init_weight(["time_text_embed.timestep_embedder.linear_1"], self.time_text_embed.timestep_embedder.linear1)
        self.init_weight(["time_text_embed.timestep_embedder.linear_2"], self.time_text_embed.timestep_embedder.linear2)

        self.txt_norm.weight = self.init_weight(['txt_norm.weight'])

        self.init_weight(['img_in'], self.img_in)
        self.init_weight(['txt_in'], self.txt_in)

        return
    
    def _post_part_loading(self):
        self.init_weight(["norm_out.linear"], self.norm_out.linear)
        self.init_weight(["proj_out"], self.proj_out)
        return

    def _major_parts_loading(self):
        
        for i in range(len(self.transformer_blocks)):
            #others
            self.init_weight([f"transformer_blocks.{i}.img_mod.1"], self.transformer_blocks[i].img_mod_proj, self.quant_dtype if self.quant_img_txt_mod else None)
            self.init_weight([f"transformer_blocks.{i}.txt_mod.1"], self.transformer_blocks[i].txt_mod_proj, self.quant_dtype if self.quant_img_txt_mod else None)

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
            self.init_weight([f"transformer_blocks.{i}.img_mlp.net.0.proj"], self.transformer_blocks[i].img_mlp.act_fn.proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.img_mlp.net.2"], self.transformer_blocks[i].img_mlp.ff_out_proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.txt_mlp.net.0.proj"], self.transformer_blocks[i].txt_mlp.act_fn.proj, self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.txt_mlp.net.2"], self.transformer_blocks[i].txt_mlp.ff_out_proj, self.quant_dtype)
        
        return
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`QwenTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
                Mask of the input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            attention_kwargs (`dict`, *optional*):
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
            if encoder_hidden_states_mask is not None:
                encoder_hidden_states_mask = encoder_hidden_states_mask.to(self.device)
            if timestep is not None:
                timestep = timestep.to(self.device)

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        hidden_states = self.img_in.forward(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm.forward(encoder_hidden_states)
        encoder_hidden_states = self.txt_in.forward(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed.forward(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed.forward(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed.forward(img_shapes, txt_seq_lens, device=hidden_states.device)

        #merge image and text rotary embeddings to one tensor
        img_freqs, txt_freqs = image_rotary_emb
        img_freqs = torch.cat([img_freqs.real, img_freqs.imag],dim=-1).to(hidden_states.dtype)
        txt_freqs = torch.cat([txt_freqs.real, txt_freqs.imag],dim=-1).to(encoder_hidden_states.dtype)
        image_rotary_emb = torch.cat([txt_freqs, img_freqs], dim=0).contiguous()

        if self.enable_caching:
            # get current step
            current_step = self.cache_config.current_steps_callback() if self.cache_config.current_steps_callback() is not None else 0

            inp = hidden_states.clone()
            inp1 = encoder_hidden_states.clone()
            temb_ = temb.clone()

            img_mod_params = self.transformer_blocks[0].img_mod_proj.forward(F.silu(temb_))
            txt_mod_params = self.transformer_blocks[0].txt_mod_proj.forward(F.silu(temb_))
            # Split modulation parameters for norm1 and norm2
            img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
            txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
            # Process image stream - norm1 + modulation
            img_normed = F.layer_norm(inp, (self.transformer_blocks[0].dim,), eps=self.transformer_blocks[0].eps)
            img_modulated, img_gate1 = self.transformer_blocks[0]._modulate(img_normed, img_mod1)
            # Process text stream - norm1 + modulation
            txt_normed = F.layer_norm(inp1, (self.transformer_blocks[0].dim,), eps=self.transformer_blocks[0].eps)
            txt_modulated, txt_gate1 = self.transformer_blocks[0]._modulate(txt_normed, txt_mod1)

            modulated_inp = txt_modulated.clone()

            if self.cache_config.negtive_cache:
                cache_key = None
                for k in self.cache_status:
                    if self.cache_status[k] and cache_key is None:
                        cache_key = k
                    self.cache_status[k] = not self.cache_status[k]
            else:
                cache_key = "positive"

            if current_step == 0:
                should_calc = True
                self.accumulated_rel_l1_distance_dict[cache_key] = 0
            else: 
                coefficients = self.coefficients[cache_key]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance_dict[cache_key] += rescale_func(((modulated_inp-self.previous_modulated_input_dict[cache_key]).abs().mean() / self.previous_modulated_input_dict[cache_key].abs().mean()).cpu().item())

                if self.accumulated_rel_l1_distance_dict[cache_key] < self.cache_config.threshold:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance_dict[cache_key] = 0
            self.previous_modulated_input_dict[cache_key] = modulated_inp

        if self.enable_caching:
            if not should_calc:
                hidden_states += self.previous_residual_dict[cache_key]
                encoder_hidden_states += self.previous_encoder_residual_dict[cache_key]
            else:
                ori_hidden_states = hidden_states.clone()
                ori_encoder_hidden_states = encoder_hidden_states.clone()
                for index_block, block in enumerate(self.transformer_blocks):
                # for index_block, block in enumerate(self.transformer_blocks[1:]):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                self.previous_residual_dict[cache_key] = hidden_states - ori_hidden_states
                self.previous_encoder_residual_dict[cache_key] = encoder_hidden_states - ori_encoder_hidden_states
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                encoder_hidden_states, hidden_states = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out.forward(hidden_states, temb)
        output = self.proj_out.forward(hidden_states)

        if self.need_resolve_oom:
            output = output.to("cpu")

        return (output,)