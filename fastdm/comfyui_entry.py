import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
import math
from einops import rearrange, repeat

from fastdm.model.sdxl import SDXLUNetModelCore
from fastdm.model.controlnets import SdxlControlNetModelCore,FluxControlNetModelCore
from fastdm.model.flux import FluxTransformer2DModelCore
from fastdm.model.sd35 import SD3TransformerModelCore
from fastdm.model.qwenimage import QwenImageTransformer2DModelCore

import numpy as np

def pad_to_patch_size(img, patch_size=(2, 2), padding_mode="circular"):
    if padding_mode == "circular" and (torch.jit.is_tracing() or torch.jit.is_scripting()):
        padding_mode = "reflect"

    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad

    return torch.nn.functional.pad(img, pad, mode=padding_mode)

class ComfyUIUNetForwardWrapper(nn.Module):
    def __init__(self, model: SDXLUNetModelCore, config):
        super(ComfyUIUNetForwardWrapper, self).__init__()
        self.model = model
        # self.dtype = next(model.parameters()).dtype
        self.dtype = torch.float16
        self.config = config
    
    def forward(
        self, x, timestep, context, y, control=None, transformer_options={}, **kwargs
    ):
        sample = x
        timesteps = timestep
        encoder_hidden_states = context
        transformer_options["transformer_index"] = 0

        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.model.time_proj.forward(timesteps).to(dtype=sample.dtype)

        if self.model.config.is_ip_adapter:
            # patch = transformer_options["patches_replace"]["attn2"][("input",4,0)].kwargs
            # image_embeds = patch[0]["cond"]
            # image_neg_embeds = patch[0]["uncond"]
            transformer_options["is_mid"] = False
            encoder_hidden_states = (encoder_hidden_states, None, None)

        emb = self.model.time_embedding.forward(t_emb)

        if y is not None:
            assert y.shape[0] == sample.shape[0]
            aug_emb = self.model.add_embedding.forward(y)

        emb = emb + aug_emb

        sample = F.conv2d(sample, self.model.conv_in_weight, self.model.conv_in_bias, 1, 1)

        is_controlnet = control is not None

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.model.down_blocks[0].forward(
            sample,
            temb=emb,
        )
        if self.model.config.is_ip_adapter:
            transformer_options["unet_block"] = ("input", 1)
        sample, [s4, s5, s6] = self.model.down_blocks[1].forward(
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            transformer_options=transformer_options,
        )
        if self.model.config.is_ip_adapter:
            transformer_options["unet_block"] = ("input", 2)
        sample, [s7, s8] = self.model.down_blocks[2].forward(
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            transformer_options=transformer_options,
        )

        mid_input = sample.clone()
        if is_controlnet:
            down_block_res_samples = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, control["output"]
            ):
                down_block_res_sample += down_block_additional_residual

        # 4. mid
        # comfyui not compute mid block for ip adapter
        if self.model.config.is_ip_adapter:
            transformer_options["unet_block"] = ("middle", 0)
        if self.model.config.is_ip_adapter:
            transformer_options["is_mid"] = True
        sample = self.model.mid_block.forward(
            mid_input, emb, encoder_hidden_states=encoder_hidden_states,transformer_options=transformer_options
        )

        if is_controlnet:
            sample = sample + control["middle"][0]

        if self.model.config.is_ip_adapter:
            transformer_options["is_mid"] = False
            
        # 5. up
        if self.model.config.is_ip_adapter:
            transformer_options["unet_block"] = ("output", 0)
        sample = self.model.up_blocks[0].forward(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
            transformer_options=transformer_options,
        )
        if self.model.config.is_ip_adapter:
            transformer_options["unet_block"] = ("output", 1)
        sample = self.model.up_blocks[1].forward(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
            transformer_options=transformer_options,
        )

        sample = self.model.up_blocks[2].forward(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
        )

        # 6. post-process
        sample = F.group_norm(sample, 32, self.model.conv_norm_out_gemma, self.model.conv_norm_out_beta, eps=1e-05)
        sample = F.silu(sample)
        sample = F.conv2d(sample, self.model.conv_out_weight, self.model.conv_out_bias, 1, 1)

        return sample
    
class ComfyUIControlnetForwardWrapper(nn.Module):
    def __init__(self, model: SdxlControlNetModelCore, config):
        super(ComfyUIControlnetForwardWrapper, self).__init__()
        self.model = model
        self.dtype = torch.float16
        self.config = config
    
    def forward(self, x, hint, timesteps, context, y=None, **kwargs):
        sample = x
        timestep = timesteps
        encoder_hidden_states = context
        class_labels = y
        controlnet_cond = hint

        # check channel order
        channel_order = self.model.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # t_emb = self.model.time_proj.forward(timesteps)
        t_emb = self.timestep_embedding(timesteps, 320, repeat_only=False).to(sample.dtype)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.model.time_embedding.forward(t_emb)
        aug_emb = None

        if self.model.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.model.config.class_embed_type == "timestep":
                class_labels = self.model.time_proj.forward(class_labels)

            class_emb = self.model.class_embedding.forward(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        if class_labels is not None:
            assert y.shape[0] == x.shape[0]
            class_emb = self.model.add_embedding.forward(class_labels)
            emb = emb + class_emb

        # 2. pre-process
        sample = F.conv2d(sample, self.model.conv_in_weight, self.model.conv_in_bias, padding=self.model.conv_in_padding)

        controlnet_cond = self.model.controlnet_cond_embedding.forward(controlnet_cond)
        sample = sample + controlnet_cond
        
        # 3. down
        down_block_res_samples = [sample,]
        for downsample_block in self.model.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block.forward(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample, res_samples = downsample_block.forward(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.model.mid_block is not None:
            if hasattr(self.model.mid_block, "has_cross_attention") and self.model.mid_block.has_cross_attention:
                sample = self.model.mid_block.forward(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample = self.model.mid_block.forward(sample, emb)

        # 5. Control net blocks
        controlnet_down_block_res_samples = []

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.model.controlnet_down_blocks):
            down_block_res_sample = F.conv2d(down_block_res_sample, controlnet_block[0], controlnet_block[1])
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + [down_block_res_sample,]

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = F.conv2d(sample, self.model.controlnet_mid_block_weight, self.model.controlnet_mid_block_bias)

        return {"middle": [mid_block_res_sample], "output": down_block_res_samples}

    def timestep_embedding(self, timesteps, dim, max_period=10000, repeat_only=False):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        if not repeat_only:
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
            )
            args = timesteps[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        else:
            embedding = repeat(timesteps, 'b -> b d', d=dim)
        return embedding

class ComfyUIFluxForwardWrapper(nn.Module):
    def __init__(self, model: FluxTransformer2DModelCore, config):
        super(ComfyUIFluxForwardWrapper, self).__init__()
        self.model = model
        # self.dtype = next(model.parameters()).dtype
        self.dtype = torch.bfloat16
        self.config = config
        self.patch_size = 2
    
    def forward(
        self, x, timestep, context, y, guidance, control=None, transformer_options={}, **kwargs
):
        # get current step
        all_steps_sigmas = transformer_options["sample_sigmas"]
        current_steps_sigmas = transformer_options["sigmas"]
        self.model.cache.config.current_steps_callback = lambda: (all_steps_sigmas == current_steps_sigmas).nonzero().item()
        self.model.cache.config.total_steps_callback = lambda: all_steps_sigmas.shape[0]

        # pre-process for input
        # ref https://github.com/comfyanonymous/ComfyUI/blob/3d2e3a6f29670809aa97b41505fa4e93ce11b98d/comfy/ldm/flux/model.py#L191
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0)
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # copy from fastdm flux.py, change controlnet sample for comfyui 
        hidden_states = img
        encoder_hidden_states = context
        pooled_projections = y
        joint_attention_kwargs = {}

        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        # hidden_states = self.x_embedder(hidden_states)
        hidden_states = self.model.x_embedder.forward(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.model.time_text_embed.forward(timestep, pooled_projections)
            if guidance is None
            else self.model.time_text_embed.forward(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.model.context_embedder.forward(encoder_hidden_states)
        # encoder_hidden_states = matmul(encoder_hidden_states, self.model.context_embedder_weight, bias_tensor=self.model.context_embedder_bias)

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
        image_rotary_emb = self.model.pos_embed.forward(ids)
        #merge cos and sin to single tensor, to ultilize the custom-rope-emb-ops
        image_rotary_emb = torch.cat((image_rotary_emb[0][:,0::2], image_rotary_emb[1][:,1::2]), dim=-1).to(hidden_states.dtype)

        if self.model.enable_caching:
            if control is not None:
                controlnet_block_samples = control.get("input") or None
                controlnet_single_block_samples = control.get("output") or None
            else:
                controlnet_block_samples = None
                controlnet_single_block_samples = None
            hidden_states = self.model.cache.apply_cache(
                model_type="flux",
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=joint_attention_kwargs,
                transformer_blocks=self.model.transformer_blocks,
                single_transformer_blocks=self.model.single_transformer_blocks,
                controlnet_block_samples=controlnet_block_samples,
                controlnet_single_block_samples=controlnet_single_block_samples,
                controlnet_blocks_repeat=False
            )
        else:
            for index_block, block in enumerate(self.model.transformer_blocks):
                encoder_hidden_states, hidden_states = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if control is not None:
                    control_i = control.get("input")
                    if index_block < len(control_i):
                        add = control_i[index_block]
                        if add is not None:
                            hidden_states += add

            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for index_block, block in enumerate(self.model.single_transformer_blocks):
                hidden_states = block.forward(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

                # controlnet residual
                if control is not None:
                    control_o = control.get("output")
                    if index_block < len(control_o):
                        add = control_o[index_block]
                        if add is not None:
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...] += add

            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.model.norm_out.forward(hidden_states, temb)
        output = self.model.proj_out.forward(hidden_states)
        # output = matmul(hidden_states, self.model.proj_out_weight, bias_tensor=self.model.proj_out_bias)

        return rearrange(output, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2)[:,:,:h,:w]
    
class ComfyUIFluxControlnetForwardWrapper(nn.Module):
    def __init__(self, model: FluxControlNetModelCore, config):
        super(ComfyUIFluxControlnetForwardWrapper, self).__init__()
        self.model = model
        self.dtype = torch.bfloat16
        self.config = config
        self.latent_input=True
        self.main_model_double = 19 # transformer double block numbers
        self.main_model_single = 38 # transformer single block numbers

    def forward(self, x, timesteps, context, y, guidance=None, hint=None, **kwargs):
        patch_size = 2
        if self.latent_input:
            hint = pad_to_patch_size(hint, (patch_size, patch_size))

        hint = rearrange(hint, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        bs, c, h, w = x.shape
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)

        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)
        img_ids = torch.zeros((h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)

        # comfyui forward input transfer to fastdm forward input
        # img:hidden_states, context:encoder_hidden_states, hint:controlnet_cond, y:pooled_projections

        controlnet_output = self.model.forward(
            hidden_states=img,
            controlnet_cond=hint,
            encoder_hidden_states=context,
            pooled_projections=y,
            timestep=timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs={},
        )
        # post-processs, to make result compatible with ComfyUI
        controlnet_double = controlnet_output[0]
        controlnet_single = controlnet_output[1]
        repeat_num = math.ceil(self.main_model_double / len(controlnet_double))
        if self.latent_input:
            out_input = ()
            for x in controlnet_double:
                    out_input += (x,) * repeat_num
        else:
            out_input = (controlnet_double*repeat_num)

        out = {"input": out_input[:self.main_model_double]}

        if controlnet_single is not None and len(controlnet_single) > 0:
            repeat_num = math.ceil(self.main_model_single / len(controlnet_single))
            out_output = ()
            if self.latent_input:
                for x in controlnet_single:
                    out_output += (x,) * repeat_num
            else:
                out_output = (controlnet_single * repeat_num)
            out["output"] = out_output[:self.main_model_single]
        return out
    
class ComfyUISD35ForwardWrapper(nn.Module):
    def __init__(self, model: SD3TransformerModelCore, config):
        super(ComfyUISD35ForwardWrapper, self).__init__()
        self.model = model
        # self.dtype = next(model.parameters()).dtype
        self.dtype = torch.bfloat16
        self.config = config
        self.patch_size = 2

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        control = None,
        transformer_options = {},
        **kwargs,
    ) -> torch.Tensor:
        """
        Variable mapping between Comfyui and fastdm
        x: hidden_states
        timesteps: timestep
        context: encoder_hidden_states
        y: pooled_projections

        """
        hidden_states = x
        timestep = timesteps
        encoder_hidden_states = context
        pooled_projections = y

        height, width = hidden_states.shape[-2:]
        
        # set current steps callback for fastdm cache
        all_steps_sigmas = transformer_options["sample_sigmas"]
        current_steps_sigmas = transformer_options["sigmas"]
        self.model.cache.config.current_steps_callback = lambda: (all_steps_sigmas == current_steps_sigmas).nonzero().item()
        
        output = self.model.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep
        )
        
        return output[0][:,:,:height,:width]

class ComfyUIQwenImageForwardWrapper(nn.Module):
    def __init__(self, model: QwenImageTransformer2DModelCore, config):
        super(ComfyUIQwenImageForwardWrapper, self).__init__()
        self.model = model
        # self.dtype = next(model.parameters()).dtype
        self.dtype = torch.bfloat16
        self.config = config
        self.patch_size = 2

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        bs, c, t, h, w = x.shape
        patch_size = self.patch_size
        hidden_states = pad_to_patch_size(x, (1, self.patch_size, self.patch_size))
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(orig_shape[0], orig_shape[1], orig_shape[-2] // 2, 2, orig_shape[-1] // 2, 2)
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5)
        hidden_states = hidden_states.reshape(orig_shape[0], (orig_shape[-2] // 2) * (orig_shape[-1] // 2), orig_shape[1] * 4)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        img_ids = torch.zeros((h_len, w_len, 3), device=x.device)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=h_len, device=x.device, dtype=x.dtype).unsqueeze(1) - (h_len // 2)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=w_len, device=x.device, dtype=x.dtype).unsqueeze(0) - (w_len // 2)

        img_shapes = [[(1, h_len, w_len)]] * bs
        return hidden_states, repeat(img_ids, "h w c -> b (h w) c", b=bs), orig_shape, img_shapes

    def forward(
        self,
        x,
        timesteps,
        context,
        attention_mask=None,
        guidance: torch.Tensor = None,
        ref_latents=None,
        transformer_options={},
        **kwargs
    ):
        """
        Variable mapping between Comfyui and fastdm
        x: hidden_states
        timesteps: timestep
        context: encoder_hidden_states
        y: pooled_projections

        """

        timestep = timesteps
        encoder_hidden_states = context
        encoder_hidden_states_mask = attention_mask
        hidden_states, img_ids, orig_shape, img_shapes = self.process_img(x)
        num_embeds = hidden_states.shape[1]
        txt_seq_lens = [context.shape[1]]

        # set current steps callback for fastdm cache
        all_steps_sigmas = transformer_options["sample_sigmas"]
        current_steps_sigmas = transformer_options["sigmas"]
        self.model.cache.config.current_steps_callback = lambda: (all_steps_sigmas == current_steps_sigmas).nonzero().item()
        
        output = self.model.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=guidance,
        )
        hidden_states = output[0]
        hidden_states = hidden_states[:, :num_embeds].view(orig_shape[0], orig_shape[-2] // 2, orig_shape[-1] // 2, orig_shape[1], 2, 2)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5)
        return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]
