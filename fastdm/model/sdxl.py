
# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py

from typing import Any, Dict
from collections import namedtuple

import torch
import torch.nn.functional as F

from fastdm.model.basemodel import BaseModelCore
from fastdm.layer.embeddings import Timesteps, TimestepEmbedding, FastdmIPAdapterPlusImageProjection, FastdmImageProjection,FastdmMultiIPAdapterImageProjection
from fastdm.layer.unetblock import DownBlock2D, CrossAttnDownBlock2D, CrossAttnUpBlock2D, UpBlock2D, UNetMidBlock2DCrossAttn

class SDXLUNetModelCore(BaseModelCore):
    def __init__(
        self, 
        sample_size=128,
        in_channels=4,
        addition_time_embed_dim=256,
        time_cond_proj_dim=None,
        is_ip_adapter=False, 
        ip_adapter_scale=0.6, 
        diffu_ipadapter_encoder_hid_proj=None, 
        data_type = torch.float16,
        quant_dtype: torch.dtype = torch.float8_e4m3fn):
        super().__init__(type="Unet")
        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels addition_time_embed_dim sample_size time_cond_proj_dim is_ip_adapter ip_adapter_scale"
        )
        self.config.in_channels = in_channels
        self.config.addition_time_embed_dim = addition_time_embed_dim
        self.config.sample_size = sample_size
        self.config.time_cond_proj_dim = time_cond_proj_dim
        self.config.is_ip_adapter = is_ip_adapter
        self.config.ip_adapter_scale = ip_adapter_scale

        self.weight_quantized = False
        self.act_quantized_static = False
        self.mixed_precision= False
        self.use_fp8 = False
        self.fp8_act_range_file = None
        self.diffu_ipadapter_encoder_hid_proj = diffu_ipadapter_encoder_hid_proj
        self.quant_asymmetric = False
        self.quant_dtype = quant_dtype

        self.conv_in_weight = torch.empty((320,4,3,3), dtype = data_type)
        self.conv_in_bias = torch.empty((320), dtype = data_type)
        self.time_proj = Timesteps(320, True, 0)
        self.time_embedding = TimestepEmbedding(in_features=320, out_features=1280, data_type=data_type)
        self.add_time_proj = Timesteps(256, True, 0)
        self.add_embedding = TimestepEmbedding(in_features=2816, out_features=1280, data_type=data_type)

        # self.encoder_hid_proj = FastdmMultiIPAdapterImageProjection(image_embed_dim=1280, cross_attention_dim=2048, num_image_text_embeds=4, data_type=data_type) if is_ip_adapter else None
        if is_ip_adapter and self.diffu_ipadapter_encoder_hid_proj is not None:
            self.encoder_hid_proj =FastdmMultiIPAdapterImageProjection(self.convert_diffusers_ipadp_image_proj_to_fastdm(self.diffu_ipadapter_encoder_hid_proj) if is_ip_adapter else None)
        else:
            self.encoder_hid_proj = None
            
        self.down_blocks = [
                DownBlock2D(in_channels=320, out_channels=320, data_type=data_type),
                CrossAttnDownBlock2D(
                    in_channels=320, 
                    out_channels=640, 
                    n_layers=2, 
                    has_ipadpt=is_ip_adapter, 
                    ipadp_scale=ip_adapter_scale,
                    data_type=data_type
                ),
                CrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,
                    has_ipadpt=is_ip_adapter, 
                    ipadp_scale=ip_adapter_scale,
                    data_type=data_type
                )
            ]
        self.up_blocks = [
                CrossAttnUpBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    prev_output_channel=1280,
                    n_layers=10,
                    has_ipadpt=is_ip_adapter, 
                    ipadp_scale=ip_adapter_scale,
                    data_type=data_type
                ),
                CrossAttnUpBlock2D(
                    in_channels=320,
                    out_channels=640,
                    prev_output_channel=1280,
                    n_layers=2,
                    has_ipadpt=is_ip_adapter, 
                    ipadp_scale=ip_adapter_scale,
                    data_type=data_type
                ),
                UpBlock2D(in_channels=320, out_channels=320, prev_output_channel=640,data_type=data_type),
            ]
        self.mid_block = UNetMidBlock2DCrossAttn(
            1280, 
            has_ipadpt=is_ip_adapter, 
            ipadp_scale=ip_adapter_scale, 
            data_type=data_type)
        self.conv_norm_out_gemma = torch.empty((320),dtype = data_type)
        self.conv_norm_out_beta = torch.empty((320),dtype = data_type)
        self.conv_out_weight = torch.empty((4,320,3,3), dtype = data_type)
        self.conv_out_bias = torch.empty((4), dtype = data_type)

    def convert_diffusers_ipadp_image_proj_to_fastdm(self, diffu_ipadapter_encoder_hid_proj):
        fastdm_ipadapter_encoder_hid_proj = None
        from diffusers.models.embeddings import IPAdapterPlusImageProjection, ImageProjection
        diffu_ipadapter_encoder_hid_proj = diffu_ipadapter_encoder_hid_proj.image_projection_layers[0]
        if isinstance(diffu_ipadapter_encoder_hid_proj, ImageProjection):
            fastdm_ipadapter_encoder_hid_proj = FastdmImageProjection(
                image_embed_dim=1280, 
                cross_attention_dim=2048, 
                num_image_text_embeds=4, 
                data_type=self.data_type
            )
        elif isinstance(diffu_ipadapter_encoder_hid_proj, IPAdapterPlusImageProjection):
            fastdm_ipadapter_encoder_hid_proj = FastdmIPAdapterPlusImageProjection(
                embed_dims=1280,
                output_dims=2048,
                hidden_dims=1280,
                depth=4,
                dim_head=64,
                heads=20,
                num_queries=16,
                ffn_ratio=4,
                data_type=self.data_type,
            )
        return fastdm_ipadapter_encoder_hid_proj



    def process_encoder_hidden_states(
        self, encoder_hidden_states: torch.Tensor, added_cond_kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
            encoder_hidden_states = self.encoder_hid_proj.forward(encoder_hidden_states)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
            # Kandinsky 2.1 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2 - style
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )
            image_embeds = added_cond_kwargs.get("image_embeds")
            encoder_hidden_states = self.encoder_hid_proj(image_embeds)
        elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "ip_image_proj":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError(
                    f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'ip_image_proj' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
                )

            if hasattr(self, "text_encoder_hid_proj") and self.text_encoder_hid_proj is not None:
                encoder_hidden_states = self.text_encoder_hid_proj.forward(encoder_hidden_states)

            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj.forward(image_embeds)
            encoder_hidden_states = (encoder_hidden_states, image_embeds)
        return encoder_hidden_states


    def _pre_part_loading(self):
        self.conv_in_weight = self.init_weight(["conv_in.weight"])
        self.conv_in_bias = self.init_weight(["conv_in.bias"])
        self.init_weight(["time_embedding.linear_1"], self.time_embedding.linear1,self.quant_dtype)
        self.init_weight(["time_embedding.linear_2"], self.time_embedding.linear2,self.quant_dtype)
        self.init_weight(["add_embedding.linear_1"], self.add_embedding.linear1,self.quant_dtype)
        self.init_weight(["add_embedding.linear_2"], self.add_embedding.linear2,self.quant_dtype)
        return
    
    def __down_block_loading(self):
        #down-block 0
        self.down_blocks[0].downsample_conv_weight = self.init_weight(["down_blocks.0.downsamplers.0.conv.weight"])
        self.down_blocks[0].downsample_conv_bias = self.init_weight(["down_blocks.0.downsamplers.0.conv.bias"])
        for i in range(2):
            self.down_blocks[0].resnets[i].norm1_gamma = self.init_weight([f"down_blocks.0.resnets.{i}.norm1.weight"])
            self.down_blocks[0].resnets[i].norm1_beta = self.init_weight([f"down_blocks.0.resnets.{i}.norm1.bias"])
            self.down_blocks[0].resnets[i].conv1_weight = self.init_weight([f"down_blocks.0.resnets.{i}.conv1.weight"])
            self.down_blocks[0].resnets[i].conv1_bias = self.init_weight([f"down_blocks.0.resnets.{i}.conv1.bias"])
            self.init_weight([f"down_blocks.0.resnets.{i}.time_emb_proj"], self.down_blocks[0].resnets[i].time_emb_proj)
            self.down_blocks[0].resnets[i].norm2_gamma = self.init_weight([f"down_blocks.0.resnets.{i}.norm2.weight"])
            self.down_blocks[0].resnets[i].norm2_beta = self.init_weight([f"down_blocks.0.resnets.{i}.norm2.bias"])
            self.down_blocks[0].resnets[i].conv2_weight = self.init_weight([f"down_blocks.0.resnets.{i}.conv2.weight"])
            self.down_blocks[0].resnets[i].conv2_bias = self.init_weight([f"down_blocks.0.resnets.{i}.conv2.bias"])
            
        for m in range(1,3):
            transformer_block_num = 2 if m==1 else 10
            if 1 == m:
                self.down_blocks[m].downsample_conv_weight = self.init_weight([f"down_blocks.{m}.downsamplers.0.conv.weight"])
                self.down_blocks[m].downsample_conv_bias = self.init_weight([f"down_blocks.{m}.downsamplers.0.conv.bias"])
            for i in range(2):
                #attention
                self.down_blocks[m].attentions[i].norm_gamma = self.init_weight([f"down_blocks.{m}.attentions.{i}.norm.weight"])
                self.down_blocks[m].attentions[i].norm_beta = self.init_weight([f"down_blocks.{m}.attentions.{i}.norm.bias"])    
                self.init_weight([f"down_blocks.{m}.attentions.{i}.proj_in"], self.down_blocks[m].attentions[i].proj_in,self.quant_dtype)                
                self.init_weight([f"down_blocks.{m}.attentions.{i}.proj_out"], self.down_blocks[m].attentions[i].proj_out,self.quant_dtype)

                for j in range(transformer_block_num):
                    self.down_blocks[m].attentions[i].transformer_blocks[j].norm1_gamma = self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm1.weight"])
                    self.down_blocks[m].attentions[i].transformer_blocks[j].norm1_beta = self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm1.bias"])
                    self.down_blocks[m].attentions[i].transformer_blocks[j].norm2_gamma = self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm2.weight"])
                    self.down_blocks[m].attentions[i].transformer_blocks[j].norm2_beta = self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm2.bias"])
                    self.down_blocks[m].attentions[i].transformer_blocks[j].norm3_gamma = self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm3.weight"])
                    self.down_blocks[m].attentions[i].transformer_blocks[j].norm3_beta = self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm3.bias"])
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_q", 
                                    f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_k", 
                                    f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_v"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn1.qkv_proj,self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_out.0"], 
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn1.out_proj,self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_q"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn2.q_proj,self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_k",
                                    f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_v"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn2.kv_proj,self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_out.0"], 
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn2.out_proj,self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.ff.net.0.proj"], 
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].ff.proj1,self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.ff.net.2"], 
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].ff.proj2,self.quant_dtype)
                #resnet
                self.down_blocks[m].resnets[i].norm1_gamma = self.init_weight([f"down_blocks.{m}.resnets.{i}.norm1.weight"])
                self.down_blocks[m].resnets[i].norm1_beta = self.init_weight([f"down_blocks.{m}.resnets.{i}.norm1.bias"])
                self.down_blocks[m].resnets[i].conv1_weight = self.init_weight([f"down_blocks.{m}.resnets.{i}.conv1.weight"])
                self.down_blocks[m].resnets[i].conv1_bias = self.init_weight([f"down_blocks.{m}.resnets.{i}.conv1.bias"])
                self.init_weight([f"down_blocks.{m}.resnets.{i}.time_emb_proj"], self.down_blocks[m].resnets[i].time_emb_proj)
                self.down_blocks[m].resnets[i].norm2_gamma = self.init_weight([f"down_blocks.{m}.resnets.{i}.norm2.weight"])
                self.down_blocks[m].resnets[i].norm2_beta = self.init_weight([f"down_blocks.{m}.resnets.{i}.norm2.bias"])
                self.down_blocks[m].resnets[i].conv2_weight = self.init_weight([f"down_blocks.{m}.resnets.{i}.conv2.weight"])
                self.down_blocks[m].resnets[i].conv2_bias = self.init_weight([f"down_blocks.{m}.resnets.{i}.conv2.bias"])
                if 0 == i:
                    self.down_blocks[m].resnets[i].convshortcut_weight = self.init_weight([f"down_blocks.{m}.resnets.{i}.conv_shortcut.weight"])
                    self.down_blocks[m].resnets[i].convshortcut_bias = self.init_weight([f"down_blocks.{m}.resnets.{i}.conv_shortcut.bias"])
        return
    
    def __mid_block_loading(self):
        transformer_block_num = 10
        attention_num = 1
        restnet_num = 2
        #attention
        for i in range(attention_num):
            self.mid_block.attentions[i].norm_gamma = self.init_weight([f"mid_block.attentions.{i}.norm.weight"])
            self.mid_block.attentions[i].norm_beta = self.init_weight([f"mid_block.attentions.{i}.norm.bias"])
            self.init_weight([f"mid_block.attentions.{i}.proj_in"], self.mid_block.attentions[i].proj_in,self.quant_dtype)
            self.init_weight([f"mid_block.attentions.{i}.proj_out"], self.mid_block.attentions[i].proj_out,self.quant_dtype)
            for j in range(transformer_block_num):
                self.mid_block.attentions[i].transformer_blocks[j].norm1_gamma = self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.norm1.weight"])
                self.mid_block.attentions[i].transformer_blocks[j].norm1_beta = self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.norm1.bias"])
                self.mid_block.attentions[i].transformer_blocks[j].norm2_gamma = self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.norm2.weight"])
                self.mid_block.attentions[i].transformer_blocks[j].norm2_beta = self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.norm2.bias"])
                self.mid_block.attentions[i].transformer_blocks[j].norm3_gamma = self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.norm3.weight"])
                self.mid_block.attentions[i].transformer_blocks[j].norm3_beta = self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.norm3.bias"])
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn1.to_q", 
                                    f"mid_block.attentions.{i}.transformer_blocks.{j}.attn1.to_k", 
                                    f"mid_block.attentions.{i}.transformer_blocks.{j}.attn1.to_v"],
                                    self.mid_block.attentions[i].transformer_blocks[j].attn1.qkv_proj,self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn1.to_out.0"], 
                                self.mid_block.attentions[i].transformer_blocks[j].attn1.out_proj,self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_q"],
                                self.mid_block.attentions[i].transformer_blocks[j].attn2.q_proj,self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_k",
                                f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_v"],
                                self.mid_block.attentions[i].transformer_blocks[j].attn2.kv_proj,self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_out.0"], 
                                self.mid_block.attentions[i].transformer_blocks[j].attn2.out_proj,self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.ff.net.0.proj"], 
                                self.mid_block.attentions[i].transformer_blocks[j].ff.proj1,self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.ff.net.2"], 
                                self.mid_block.attentions[i].transformer_blocks[j].ff.proj2,self.quant_dtype)
        #resnet
        for i in range(restnet_num):
            self.mid_block.resnets[i].norm1_gamma = self.init_weight([f"mid_block.resnets.{i}.norm1.weight"])
            self.mid_block.resnets[i].norm1_beta = self.init_weight([f"mid_block.resnets.{i}.norm1.bias"])
            self.mid_block.resnets[i].conv1_weight = self.init_weight([f"mid_block.resnets.{i}.conv1.weight"])
            self.mid_block.resnets[i].conv1_bias = self.init_weight([f"mid_block.resnets.{i}.conv1.bias"])
            self.init_weight([f"mid_block.resnets.{i}.time_emb_proj"], self.mid_block.resnets[i].time_emb_proj)
            self.mid_block.resnets[i].norm2_gamma = self.init_weight([f"mid_block.resnets.{i}.norm2.weight"])
            self.mid_block.resnets[i].norm2_beta = self.init_weight([f"mid_block.resnets.{i}.norm2.bias"])
            self.mid_block.resnets[i].conv2_weight = self.init_weight([f"mid_block.resnets.{i}.conv2.weight"])
            self.mid_block.resnets[i].conv2_bias = self.init_weight([f"mid_block.resnets.{i}.conv2.bias"])
        return
    
    def __up_block_loading(self):
        block_num = 3
        for m in range(block_num):
            if 2 == m:
                attention_num = 0
            else:
                attention_num = 3
            resnet_num = 3
            if 0 == m:
                transformer_block_num = 10
            elif 1 == m:
                transformer_block_num = 2
            else:
                transformer_block_num = 0
            if 1 == m or 0 == m:
                self.up_blocks[m].upsample_conv_weight = self.init_weight([f"up_blocks.{m}.upsamplers.0.conv.weight"])
                self.up_blocks[m].upsample_conv_bias = self.init_weight([f"up_blocks.{m}.upsamplers.0.conv.bias"])
            #attention
            for i in range(attention_num):
                self.up_blocks[m].attentions[i].norm_gamma = self.init_weight([f"up_blocks.{m}.attentions.{i}.norm.weight"])
                self.up_blocks[m].attentions[i].norm_beta = self.init_weight([f"up_blocks.{m}.attentions.{i}.norm.bias"])
                self.init_weight([f"up_blocks.{m}.attentions.{i}.proj_in"], self.up_blocks[m].attentions[i].proj_in,self.quant_dtype)
                self.init_weight([f"up_blocks.{m}.attentions.{i}.proj_out"], self.up_blocks[m].attentions[i].proj_out,self.quant_dtype)

                for j in range(transformer_block_num):
                    self.up_blocks[m].attentions[i].transformer_blocks[j].norm1_gamma = self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm1.weight"])
                    self.up_blocks[m].attentions[i].transformer_blocks[j].norm1_beta = self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm1.bias"])
                    self.up_blocks[m].attentions[i].transformer_blocks[j].norm2_gamma = self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm2.weight"])
                    self.up_blocks[m].attentions[i].transformer_blocks[j].norm2_beta = self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm2.bias"])
                    self.up_blocks[m].attentions[i].transformer_blocks[j].norm3_gamma = self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm3.weight"])
                    self.up_blocks[m].attentions[i].transformer_blocks[j].norm3_beta = self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.norm3.bias"])
                    self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_q", 
                                    f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_k", 
                                    f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_v"],
                                    self.up_blocks[m].attentions[i].transformer_blocks[j].attn1.qkv_proj,self.quant_dtype)
                    self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_out.0"], 
                                    self.up_blocks[m].attentions[i].transformer_blocks[j].attn1.out_proj,self.quant_dtype)
                    self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_q"],
                                    self.up_blocks[m].attentions[i].transformer_blocks[j].attn2.q_proj,self.quant_dtype)
                    self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_k",
                                    f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_v"],
                                    self.up_blocks[m].attentions[i].transformer_blocks[j].attn2.kv_proj,self.quant_dtype)
                    self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_out.0"], 
                                    self.up_blocks[m].attentions[i].transformer_blocks[j].attn2.out_proj,self.quant_dtype)   
                    self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.ff.net.0.proj"], 
                                    self.up_blocks[m].attentions[i].transformer_blocks[j].ff.proj1,self.quant_dtype)
                    self.init_weight([f"up_blocks.{m}.attentions.{i}.transformer_blocks.{j}.ff.net.2"], 
                                    self.up_blocks[m].attentions[i].transformer_blocks[j].ff.proj2,self.quant_dtype)
            #resnets
            for i in range(resnet_num):
                #resnet
                self.up_blocks[m].resnets[i].norm1_gamma = self.init_weight([f"up_blocks.{m}.resnets.{i}.norm1.weight"])
                self.up_blocks[m].resnets[i].norm1_beta = self.init_weight([f"up_blocks.{m}.resnets.{i}.norm1.bias"])
                self.up_blocks[m].resnets[i].conv1_weight = self.init_weight([f"up_blocks.{m}.resnets.{i}.conv1.weight"])
                self.up_blocks[m].resnets[i].conv1_bias = self.init_weight([f"up_blocks.{m}.resnets.{i}.conv1.bias"])
                self.init_weight([f"up_blocks.{m}.resnets.{i}.time_emb_proj"], self.up_blocks[m].resnets[i].time_emb_proj)
                self.up_blocks[m].resnets[i].norm2_gamma = self.init_weight([f"up_blocks.{m}.resnets.{i}.norm2.weight"])
                self.up_blocks[m].resnets[i].norm2_beta = self.init_weight([f"up_blocks.{m}.resnets.{i}.norm2.bias"])
                self.up_blocks[m].resnets[i].conv2_weight = self.init_weight([f"up_blocks.{m}.resnets.{i}.conv2.weight"])
                self.up_blocks[m].resnets[i].conv2_bias = self.init_weight([f"up_blocks.{m}.resnets.{i}.conv2.bias"])
                self.up_blocks[m].resnets[i].convshortcut_weight = self.init_weight([f"up_blocks.{m}.resnets.{i}.conv_shortcut.weight"])
                self.up_blocks[m].resnets[i].convshortcut_bias = self.init_weight([f"up_blocks.{m}.resnets.{i}.conv_shortcut.bias"])

        return
    
    
    def _major_parts_loading(self):
        self.__down_block_loading()
        self.__mid_block_loading()
        self.__up_block_loading()
        return
    
    def _post_part_loading(self):
        self.conv_norm_out_gemma = self.init_weight([f"conv_norm_out.weight"])
        self.conv_norm_out_beta = self.init_weight([f"conv_norm_out.bias"])
        self.conv_out_weight = self.init_weight([f"conv_out.weight"])
        self.conv_out_bias = self.init_weight([f"conv_out.bias"])
        return
    
    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs
    ):
        # Implement the forward pass through the model
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj.forward(timesteps).to(dtype=sample.dtype)

        emb = self.time_embedding.forward(t_emb)

        text_embeds = added_cond_kwargs.get("text_embeds")
        time_ids = added_cond_kwargs.get("time_ids")

        if self.config.is_ip_adapter:
            image_embeds = added_cond_kwargs.get("image_embeds")
            image_embeds = self.encoder_hid_proj.forward(image_embeds)
            neg_image_embeds = added_cond_kwargs.get("neg_image_embeds")
            encoder_hidden_states = (encoder_hidden_states, image_embeds,neg_image_embeds)

        time_embeds = self.add_time_proj.forward(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = self.add_embedding.forward(add_embeds)

        emb = emb + aug_emb

        sample = F.conv2d(sample, self.conv_in_weight, self.conv_in_bias, 1, 1)

        is_controlnet = kwargs.get("mid_block_additional_residual") is not None and kwargs.get("down_block_additional_residuals") is not None

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0].forward(
            sample,
            temb=emb,
        )

        sample, [s4, s5, s6] = self.down_blocks[1].forward(
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        sample, [s7, s8] = self.down_blocks[2].forward(
            sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
        )

        if is_controlnet:
            down_block_res_samples = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, kwargs.get("down_block_additional_residuals")
            ):
                down_block_res_sample += down_block_additional_residual

        # 4. mid
        sample = self.mid_block.forward(
            sample, emb, encoder_hidden_states=encoder_hidden_states
        )

        if is_controlnet:
            sample = sample + kwargs.get("mid_block_additional_residual")

        # 5. up
        sample = self.up_blocks[0].forward(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s6, s7, s8],
            encoder_hidden_states=encoder_hidden_states,
        )

        sample = self.up_blocks[1].forward(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s3, s4, s5],
            encoder_hidden_states=encoder_hidden_states,
        )

        sample = self.up_blocks[2].forward(
            hidden_states=sample,
            temb=emb,
            res_hidden_states_tuple=[s0, s1, s2],
        )

        # 6. post-process
        sample = F.group_norm(sample, 32, self.conv_norm_out_gemma, self.conv_norm_out_beta, eps=1e-05)
        sample = F.silu(sample)
        sample = F.conv2d(sample, self.conv_out_weight, self.conv_out_bias, 1, 1)

        return [sample]
    