
# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py

from typing import Any, Dict, List, Optional, Tuple, Union
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F

from fastdm.layer.embeddings import Timesteps, TimestepEmbedding, TextImageProjection, TextTimeEmbedding, TextImageTimeEmbedding
from fastdm.layer.unetblock import DownBlock2D, CrossAttnDownBlock2D, UNetMidBlock2DCrossAttn
from fastdm.layer.embeddings import FluxPosEmbed, CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings
from fastdm.model.flux import FluxTransformerBlock, FluxSingleTransformerBlock
from fastdm.layer.qlinear import QLinear
from fastdm.model.basemodel import BaseModelCore

class ControlNetConditioningEmbedding:
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        data_type: torch.dtype = torch.float16
    ):
        super().__init__()

        #self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)
        self.conv_in_weight = torch.empty((block_out_channels[0], conditioning_channels, 3, 3), dtype = data_type)
        self.conv_in_bias = torch.empty((block_out_channels[0]), dtype = data_type)

        self.blocks = []

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            # self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            # self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))
            self.blocks.append(
                [torch.empty((channel_in, channel_in, 3, 3), dtype = data_type),
                 torch.empty((channel_in), dtype = data_type)]
                            )
            self.blocks.append(
                [torch.empty((channel_out, channel_in, 3, 3), dtype = data_type),
                 torch.empty((channel_out), dtype = data_type)]
                            )

        # self.conv_out = zero_module(
        #     nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        # )
        self.conv_out_weight = torch.empty((conditioning_embedding_channels, block_out_channels[-1], 3, 3), dtype = data_type)
        self.conv_out_bias = torch.empty((conditioning_embedding_channels), dtype = data_type)

    def forward(self, conditioning):
        #embedding = self.conv_in(conditioning)
        embedding = F.conv2d(conditioning, self.conv_in_weight, self.conv_in_bias, padding=1)
        embedding = F.silu(embedding)

        for i, block in enumerate(self.blocks):
            #embedding = block(embedding)
            embedding = F.conv2d(embedding, block[0], block[1], stride=1 if 0 == (i%2) else 2, padding=1)
            embedding = F.silu(embedding)

        #embedding = self.conv_out(embedding)
        embedding = F.conv2d(embedding, self.conv_out_weight, self.conv_out_bias, padding=1)

        return embedding


class SdxlControlNetModelCore(BaseModelCore):
    """
    A ControlNet model.

    Args:
        in_channels (`int`, defaults to 4):
            The number of channels in the input sample.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        freq_shift (`int`, defaults to 0):
            The frequency shift to apply to the time embedding.
        down_block_types (`tuple[str]`, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        only_cross_attention (`Union[bool, Tuple[bool]]`, defaults to `False`):
        block_out_channels (`tuple[int]`, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, defaults to 2):
            The number of layers per block.
        downsample_padding (`int`, defaults to 1):
            The padding to use for the downsampling convolution.
        mid_block_scale_factor (`float`, defaults to 1):
            The scale factor to use for the mid block.
        act_fn (`str`, defaults to "silu"):
            The activation function to use.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the normalization. If None, normalization and activation layers is skipped
            in post-processing.
        norm_eps (`float`, defaults to 1e-5):
            The epsilon to use for the normalization.
        cross_attention_dim (`int`, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int` or `Tuple[int]`, *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_2d_blocks.CrossAttnDownBlock2D`], [`~models.unet_2d_blocks.CrossAttnUpBlock2D`],
            [`~models.unet_2d_blocks.UNetMidBlock2DCrossAttn`].
        encoder_hid_dim (`int`, *optional*, defaults to None):
            If `encoder_hid_dim_type` is defined, `encoder_hidden_states` will be projected from `encoder_hid_dim`
            dimension to `cross_attention_dim`.
        encoder_hid_dim_type (`str`, *optional*, defaults to `None`):
            If given, the `encoder_hidden_states` and potentially other embeddings are down-projected to text
            embeddings of dimension `cross_attention` according to `encoder_hid_dim_type`.
        attention_head_dim (`Union[int, Tuple[int]]`, defaults to 8):
            The dimension of the attention heads.
        use_linear_projection (`bool`, defaults to `False`):
        class_embed_type (`str`, *optional*, defaults to `None`):
            The type of class embedding to use which is ultimately summed with the time embeddings. Choose from None,
            `"timestep"`, `"identity"`, `"projection"`, or `"simple_projection"`.
        addition_embed_type (`str`, *optional*, defaults to `None`):
            Configures an optional embedding which will be summed with the time embeddings. Choose from `None` or
            "text". "text" will use the `TextTimeEmbedding` layer.
        num_class_embeds (`int`, *optional*, defaults to 0):
            Input dimension of the learnable embedding matrix to be projected to `time_embed_dim`, when performing
            class conditioning with `class_embed_type` equal to `None`.
        upcast_attention (`bool`, defaults to `False`):
        resnet_time_scale_shift (`str`, defaults to `"default"`):
            Time scale shift config for ResNet blocks (see `ResnetBlock2D`). Choose from `default` or `scale_shift`.
        projection_class_embeddings_input_dim (`int`, *optional*, defaults to `None`):
            The dimension of the `class_labels` input when `class_embed_type="projection"`. Required when
            `class_embed_type="projection"`.
        controlnet_conditioning_channel_order (`str`, defaults to `"rgb"`):
            The channel order of conditional image. Will convert to `rgb` if it's `bgr`.
        conditioning_embedding_out_channels (`tuple[int]`, *optional*, defaults to `(16, 32, 96, 256)`):
            The tuple of output channel for each block in the `conditioning_embedding` layer.
        global_pool_conditions (`bool`, defaults to `False`):
            TODO(Patrick) - unused parameter.
        addition_embed_type_num_heads (`int`, defaults to 64):
            The number of heads to use for the `TextTimeEmbedding` layer.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D"
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int, ...] = (320, 640, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 2048,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = (1, 2, 10),
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int, ...]] = (5,10,20),
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = True,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = "text_time",
        addition_time_embed_dim: Optional[int] = 256,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = 2816,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
        data_type = torch.float16,
        quant_dtype: torch.dtype = torch.float8_e4m3fn
    ):
        super().__init__(type="Unet")

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "controlnet_conditioning_channel_order class_embed_type addition_embed_type global_pool_conditions"
        )
        self.config.controlnet_conditioning_channel_order = controlnet_conditioning_channel_order
        self.config.class_embed_type = class_embed_type
        self.config.addition_embed_type = addition_embed_type
        self.config.global_pool_conditions = global_pool_conditions

        self.quant_dtype = quant_dtype

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        self.controlnet_conditioning_channel_order = controlnet_conditioning_channel_order

        # input
        conv_in_kernel = 3
        self.conv_in_padding = (conv_in_kernel - 1) // 2
        # self.conv_in = nn.Conv2d(
        #     in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        # )
        self.conv_in_weight = torch.empty((block_out_channels[0], in_channels, conv_in_kernel, conv_in_kernel), dtype = data_type)
        self.conv_in_bias = torch.empty((block_out_channels[0]), dtype = data_type)

        # time
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            data_type,
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = QLinear(encoder_hid_dim, cross_attention_dim, data_type=data_type)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kandinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
                data_type=data_type,
            )

        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim, dtype=data_type)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, data_type)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim, data_type)
        else:
            self.class_embedding = None

        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads,data_type=data_type
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kandinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim, data_type=data_type
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim, data_type)

        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
            data_type=data_type,
        )

        self.controlnet_down_blocks = []

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]

        self.controlnet_down_blocks.append([
            torch.empty((output_channel, output_channel, 1, 1), dtype = data_type),
            torch.empty((output_channel), dtype = data_type)
        ])

        self.down_blocks = [
                DownBlock2D(in_channels=320, out_channels=320,data_type=data_type),
                CrossAttnDownBlock2D(in_channels=320, out_channels=640, n_layers=2,data_type=data_type),
                CrossAttnDownBlock2D(
                    in_channels=640,
                    out_channels=1280,
                    n_layers=10,
                    has_downsamplers=False,
                    data_type=data_type,
                )
            ]

        for i in range(len(down_block_types)):
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            for _ in range(layers_per_block):
                self.controlnet_down_blocks.append([
                    torch.empty((output_channel, output_channel, 1, 1), dtype = data_type),
                    torch.empty((output_channel), dtype = data_type)
                ])

            if not is_final_block:
                self.controlnet_down_blocks.append([
                    torch.empty((output_channel, output_channel, 1, 1), dtype = data_type),
                    torch.empty((output_channel), dtype = data_type)
                ])

        # mid
        mid_block_channel = block_out_channels[-1]

        self.controlnet_mid_block_weight = torch.empty((mid_block_channel, mid_block_channel, 1, 1), dtype = data_type)
        self.controlnet_mid_block_bias = torch.empty((mid_block_channel), dtype = data_type)

        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(1280,data_type=data_type)
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        
        self.mixed_precision = False

    def _pre_part_loading(self):
        self.conv_in_weight = self.init_weight(["conv_in.weight"])
        self.conv_in_bias = self.init_weight(["conv_in.bias"])
        self.init_weight(["time_embedding.linear_1"], self.time_embedding.linear1, quant_dtype=self.quant_dtype)
        self.init_weight(["time_embedding.linear_2"], self.time_embedding.linear2, quant_dtype=self.quant_dtype)
        self.init_weight(["add_embedding.linear_1"], self.add_embedding.linear1, quant_dtype=self.quant_dtype)
        self.init_weight(["add_embedding.linear_2"], self.add_embedding.linear2, quant_dtype=self.quant_dtype)

        #controlnet_cond_embedding
        self.controlnet_cond_embedding.conv_in_weight = self.init_weight(["controlnet_cond_embedding.conv_in.weight"])
        self.controlnet_cond_embedding.conv_in_bias = self.init_weight(["controlnet_cond_embedding.conv_in.bias"])
        self.controlnet_cond_embedding.conv_out_weight = self.init_weight(["controlnet_cond_embedding.conv_out.weight"])
        self.controlnet_cond_embedding.conv_out_bias = self.init_weight(["controlnet_cond_embedding.conv_out.bias"])
        block_num = len(self.controlnet_cond_embedding.blocks)
        for i in range(block_num):
            self.controlnet_cond_embedding.blocks[i][0] = self.init_weight([f"controlnet_cond_embedding.blocks.{i}.weight"])
            self.controlnet_cond_embedding.blocks[i][1] = self.init_weight([f"controlnet_cond_embedding.blocks.{i}.bias"])
        return
    
    def __mid_block_loading(self):
        transformer_block_num = 10
        attention_num = 1
        restnet_num = 2
        #attention
        for i in range(attention_num):
            self.mid_block.attentions[i].norm_gamma = self.init_weight([f"mid_block.attentions.{i}.norm.weight"])
            self.mid_block.attentions[i].norm_beta = self.init_weight([f"mid_block.attentions.{i}.norm.bias"])
            self.init_weight([f"mid_block.attentions.{i}.proj_in"], self.mid_block.attentions[i].proj_in, quant_dtype=self.quant_dtype)
            self.init_weight([f"mid_block.attentions.{i}.proj_out"], self.mid_block.attentions[i].proj_out, quant_dtype=self.quant_dtype)
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
                                self.mid_block.attentions[i].transformer_blocks[j].attn1.qkv_proj, quant_dtype=self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn1.to_out.0"],
                                self.mid_block.attentions[i].transformer_blocks[j].attn1.out_proj, quant_dtype=self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_q"],self.mid_block.attentions[i].transformer_blocks[j].attn2.q_proj, quant_dtype=self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_k",
                                f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_v"],
                                self.mid_block.attentions[i].transformer_blocks[j].attn2.kv_proj, quant_dtype=self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.attn2.to_out.0"],
                                self.mid_block.attentions[i].transformer_blocks[j].attn2.out_proj, quant_dtype=self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.ff.net.0.proj"],
                                self.mid_block.attentions[i].transformer_blocks[j].ff.proj1, quant_dtype=self.quant_dtype)
                self.init_weight([f"mid_block.attentions.{i}.transformer_blocks.{j}.ff.net.2"],
                                self.mid_block.attentions[i].transformer_blocks[j].ff.proj2, quant_dtype=self.quant_dtype)
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

        #controlnet_mid_block
        self.controlnet_mid_block_weight = self.init_weight(["controlnet_mid_block.weight"])
        self.controlnet_mid_block_bias = self.init_weight(["controlnet_mid_block.bias"])
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

        #down-block 1 and down-block 2
        for m in range(1,3):
            transformer_block_num = 2 if m==1 else 10
            if 1 == m:
                self.down_blocks[m].downsample_conv_weight = self.init_weight([f"down_blocks.{m}.downsamplers.0.conv.weight"])
                self.down_blocks[m].downsample_conv_bias = self.init_weight([f"down_blocks.{m}.downsamplers.0.conv.bias"])
            for i in range(2):
                #attention
                self.down_blocks[m].attentions[i].norm_gamma = self.init_weight([f"down_blocks.{m}.attentions.{i}.norm.weight"])
                self.down_blocks[m].attentions[i].norm_beta = self.init_weight([f"down_blocks.{m}.attentions.{i}.norm.bias"])
                self.init_weight([f"down_blocks.{m}.attentions.{i}.proj_in"], self.down_blocks[m].attentions[i].proj_in, quant_dtype=self.quant_dtype)
                self.init_weight([f"down_blocks.{m}.attentions.{i}.proj_out"], self.down_blocks[m].attentions[i].proj_out, quant_dtype=self.quant_dtype)
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
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn1.qkv_proj, quant_dtype=self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn1.to_out.0"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn1.out_proj, quant_dtype=self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_q"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn2.q_proj, quant_dtype=self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_k",
                                    f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_v"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn2.kv_proj, quant_dtype=self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.attn2.to_out.0"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].attn2.out_proj, quant_dtype=self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.ff.net.0.proj"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].ff.proj1, quant_dtype=self.quant_dtype)
                    self.init_weight([f"down_blocks.{m}.attentions.{i}.transformer_blocks.{j}.ff.net.2"],
                                    self.down_blocks[m].attentions[i].transformer_blocks[j].ff.proj2, quant_dtype=self.quant_dtype)
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

        #controlnet_down_blocks
        blocks_num = len(self.controlnet_down_blocks)
        for i in range(blocks_num):
            self.controlnet_down_blocks[i][0] = self.init_weight([f"controlnet_down_blocks.{i}.weight"])
            self.controlnet_down_blocks[i][1] = self.init_weight([f"controlnet_down_blocks.{i}.bias"])

        return

    def _major_parts_loading(self):
        self.__mid_block_loading()
        self.__down_block_loading()
    
    def _post_part_loading(self):
        pass
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ):
        """
        The [`ControlNetModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor.
            timestep (`Union[torch.Tensor, float, int]`):
                The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor`):
                The encoder hidden states.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            timestep_cond (`torch.Tensor`, *optional*, defaults to `None`):
                Additional conditional embeddings for timestep. If provided, the embeddings will be summed with the
                timestep_embedding passed through the `self.time_embedding` layer to obtain the final timestep
                embeddings.
            attention_mask (`torch.Tensor`, *optional*, defaults to `None`):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            added_cond_kwargs (`dict`):
                Additional conditions for the Stable Diffusion XL UNet.
            cross_attention_kwargs (`dict[str]`, *optional*, defaults to `None`):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor`.
            guess_mode (`bool`, defaults to `False`):
                In this mode, the ControlNet encoder tries its best to recognize the input content of the input even if
                you remove all prompts. A `guidance_scale` between 3.0 and 5.0 is recommended.
            return_dict (`bool`, defaults to `True`):
                Whether or not to return a [`~models.controlnets.controlnet.ControlNetOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.controlnets.controlnet.ControlNetOutput`] **or** `tuple`:
                If `return_dict` is `True`, a [`~models.controlnets.controlnet.ControlNetOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """
        # check channel order
        channel_order = self.config.controlnet_conditioning_channel_order

        if channel_order == "rgb":
            # in rgb order by default
            ...
        elif channel_order == "bgr":
            controlnet_cond = torch.flip(controlnet_cond, dims=[1])
        else:
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

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

        t_emb = self.time_proj.forward(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding.forward(t_emb)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj.forward(class_labels)

            class_emb = self.class_embedding.forward(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding.forward(encoder_hidden_states)

            elif self.config.addition_embed_type == "text_time":
                if "text_embeds" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
                    )
                text_embeds = added_cond_kwargs.get("text_embeds")
                if "time_ids" not in added_cond_kwargs:
                    raise ValueError(
                        f"{self.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
                    )
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj.forward(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))

                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding.forward(add_embeds)

        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = F.conv2d(sample, self.conv_in_weight, self.conv_in_bias, padding=self.conv_in_padding)

        controlnet_cond = self.controlnet_cond_embedding.forward(controlnet_cond)
        sample = sample + controlnet_cond

        # 3. down
        down_block_res_samples = [sample,]
        for downsample_block in self.down_blocks:
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
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block.forward(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states
                )
            else:
                sample = self.mid_block.forward(sample, emb)

        # 5. Control net blocks

        controlnet_down_block_res_samples = []

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = F.conv2d(down_block_res_sample, controlnet_block[0], controlnet_block[1])
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + [down_block_res_sample,]

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = F.conv2d(sample, self.controlnet_mid_block_weight, self.controlnet_mid_block_bias)

        # 6. scaling
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)  # 0.1 to 1.0
            scales = scales * conditioning_scale
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            down_block_res_samples = [sample * conditioning_scale for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        return (down_block_res_samples, mid_block_res_sample)


class FluxControlNetModelCore(BaseModelCore):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 5,
        num_single_layers: int = 0,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: List[int] = [16, 56, 56],
        num_mode: int = None,
        conditioning_embedding_channels: int = None,
        data_type = torch.bfloat16,
        quant_dtype = torch.float8_e4m3fn
    ):
        super().__init__(type="DiT")
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim, data_type=data_type
        )

        self.context_embedder = QLinear(in_features=joint_attention_dim, out_features=self.inner_dim, data_type=data_type)
        self.x_embedder = QLinear(in_features=in_channels, out_features=self.inner_dim, data_type=data_type)

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

        # controlnet blocks
        self.controlnet_blocks = [
            QLinear(in_features=self.inner_dim, out_features=self.inner_dim, data_type=data_type)
            for i in range(num_layers)
        ]

        self.controlnet_single_blocks = [
            QLinear(inner_dim=self.inner_dim, out_features=self.inner_dim, data_type=data_type)
            for i in range(num_single_layers)
        ]

        self.input_hint_block = None
        self.mixed_precision = False

        self.controlnet_x_embedder = QLinear(in_features=in_channels, out_features=self.inner_dim, data_type=data_type)


    def _pre_part_loading(self):
        self.init_weight(["time_text_embed.timestep_embedder.linear_1"],self.time_text_embed.timestep_embedder.linear1)
        self.init_weight(["time_text_embed.timestep_embedder.linear_2"],self.time_text_embed.timestep_embedder.linear2)
        self.init_weight(["time_text_embed.guidance_embedder.linear_1"],self.time_text_embed.guidance_embedder.linear1)
        self.init_weight(["time_text_embed.guidance_embedder.linear_2"],self.time_text_embed.guidance_embedder.linear2)
        self.init_weight(["time_text_embed.text_embedder.linear_1"],self.time_text_embed.text_embedder.linear1)
        self.init_weight(["time_text_embed.text_embedder.linear_2"],self.time_text_embed.text_embedder.linear2)
        self.init_weight(["context_embedder"], self.context_embedder)
        self.init_weight(["x_embedder"], self.x_embedder)
        self.init_weight(["controlnet_x_embedder"], self.controlnet_x_embedder)
        return

    def __transformer_block_loading(self):
        for i in range(len(self.transformer_blocks)):
            #norm
            self.init_weight([f"transformer_blocks.{i}.norm1.linear"], self.transformer_blocks[i].norm1.linear)
            self.init_weight([f"transformer_blocks.{i}.norm1_context.linear"], self.transformer_blocks[i].norm1_context.linear)

            #attention
            self.transformer_blocks[i].attn.norm_q_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_q.weight"])
            self.transformer_blocks[i].attn.norm_k_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_k.weight"])
            self.init_weight(
                [f"transformer_blocks.{i}.attn.to_q",f"transformer_blocks.{i}.attn.to_k", f"transformer_blocks.{i}.attn.to_v"],
                self.transformer_blocks[i].attn.qkv, quant_dtype=self.quant_dtype
            )
            self.init_weight(
                [f"transformer_blocks.{i}.attn.add_k_proj", f"transformer_blocks.{i}.attn.add_v_proj", f"transformer_blocks.{i}.attn.add_q_proj"],
                self.transformer_blocks[i].attn.add_qkv_proj, quant_dtype=self.quant_dtype
            )
            self.init_weight([f"transformer_blocks.{i}.attn.to_out.0"], self.transformer_blocks[i].attn.to_out, quant_dtype=self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.attn.to_add_out"], self.transformer_blocks[i].attn.to_add_out, quant_dtype=self.quant_dtype)
            self.transformer_blocks[i].attn.norm_added_q_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_added_q.weight"])
            self.transformer_blocks[i].attn.norm_added_k_weight = self.init_weight([f"transformer_blocks.{i}.attn.norm_added_k.weight"])

            #feedforward
            self.init_weight([f"transformer_blocks.{i}.ff.net.0.proj"], self.transformer_blocks[i].ff.act_fn.proj, quant_dtype=self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.ff.net.2"], self.transformer_blocks[i].ff.ff_out_proj, quant_dtype=self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.ff_context.net.0.proj"], self.transformer_blocks[i].ff_context.act_fn.proj, quant_dtype=self.quant_dtype)
            self.init_weight([f"transformer_blocks.{i}.ff_context.net.2"], self.transformer_blocks[i].ff_context.ff_out_proj, quant_dtype=self.quant_dtype)
        return
    
    def __single_trans_block_loading(self):
        for i in range(len(self.single_transformer_blocks)):
            #norm
            self.init_weight([f"single_transformer_blocks.{i}.norm.linear"], self.single_transformer_blocks[i].norm.linear)
            #proj
            self.init_weight([f"single_transformer_blocks.{i}.proj_mlp"], self.single_transformer_blocks[i].proj_mlp, quant_dtype=self.quant_dtype)
            self.init_weight([f"single_transformer_blocks.{i}.proj_out"], self.single_transformer_blocks[i].proj_out, quant_dtype=self.quant_dtype)
            #attn
            self.single_transformer_blocks[i].attn.norm_q_weight = self.init_weight([f"single_transformer_blocks.{i}.attn.norm_q.weight"])
            self.single_transformer_blocks[i].attn.norm_k_weight = self.init_weight([f"single_transformer_blocks.{i}.attn.norm_k.weight"])
            self.init_weight(
                [f"single_transformer_blocks.{i}.attn.to_q", f"single_transformer_blocks.{i}.attn.to_k", f"single_transformer_blocks.{i}.attn.to_v"],
                self.single_transformer_blocks[i].attn.qkv, quant_dtype=self.quant_dtype
            )

        return
    
    def _major_parts_loading(self):
        self.__transformer_block_loading()
        self.__single_trans_block_loading()
        return

    def __controlnet_block_loading(self):
        for i in range(len(self.controlnet_blocks)):
            self.init_weight([f"controlnet_blocks.{i}"], self.controlnet_blocks[i], quant_dtype=self.quant_dtype)
        return

    def _post_part_loading(self):
        self.__controlnet_block_loading()
        return
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        controlnet_mode: torch.Tensor = None,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            controlnet_cond (`torch.Tensor`):
                The conditional input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            controlnet_mode (`torch.Tensor`):
                The mode tensor of shape `(batch_size, 1)`.
            conditioning_scale (`float`, defaults to `1.0`):
                The scale factor for ControlNet outputs.
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
        # current implementation controlnt not support lora

        hidden_states = self.x_embedder.forward(hidden_states)

        if self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)
            batch_size, channels, height_pw, width_pw = controlnet_cond.shape
            height = height_pw // self.config.patch_size
            width = width_pw // self.config.patch_size
            controlnet_cond = controlnet_cond.reshape(
                batch_size, channels, height, self.config.patch_size, width, self.config.patch_size
            )
            controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
            controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1)

        # add
        hidden_states = hidden_states + self.controlnet_x_embedder.forward(controlnet_cond)

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
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed.forward(ids)
        #merge cos and sin to single tensor, to ultilize the custom-rope-emb-ops
        image_rotary_emb = torch.cat((image_rotary_emb[0][:,0::2], image_rotary_emb[1][:,1::2]), dim=-1).to(hidden_states.dtype)

        block_samples = ()
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            block_samples = block_samples + (hidden_states,)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        single_block_samples = ()
        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block.forward(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            single_block_samples = single_block_samples + (hidden_states[:, encoder_hidden_states.shape[1] :],)

        # controlnet blocks
        controlnet_block_samples = ()
        for block_sample, controlnet_block in zip(block_samples, self.controlnet_blocks):
            block_sample = controlnet_block.forward(block_sample)
            controlnet_block_samples = controlnet_block_samples + (block_sample,)

        controlnet_single_block_samples = ()
        for block_sample, controlnet_block in zip(single_block_samples, self.controlnet_single_blocks):
            block_sample = controlnet_block.forward(block_sample)
            controlnet_single_block_samples = controlnet_single_block_samples + (block_sample,)

        # scaling
        controlnet_block_samples = [sample * conditioning_scale for sample in controlnet_block_samples]
        controlnet_single_block_samples = [sample * conditioning_scale for sample in controlnet_single_block_samples]

        controlnet_block_samples = None if len(controlnet_block_samples) == 0 else controlnet_block_samples
        controlnet_single_block_samples = (
            None if len(controlnet_single_block_samples) == 0 else controlnet_single_block_samples
        )
 
        return (controlnet_block_samples, controlnet_single_block_samples)
