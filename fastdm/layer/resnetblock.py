# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py

import torch
import torch.nn.functional as F

from fastdm.layer.qlinear import QLinear

class ResnetBlock2D:
    def __init__(self, in_channels, out_channels, conv_shortcut=True, data_type=torch.float16):
        super(ResnetBlock2D, self).__init__()
        self.norm1_gamma = torch.empty((in_channels), dtype = data_type)
        self.norm1_beta = torch.empty((in_channels), dtype = data_type)
        self.conv1_weight = torch.empty((out_channels, in_channels, 3, 3), dtype = data_type)
        self.conv1_bias = torch.empty((out_channels), dtype = data_type)
        # self.time_emb_proj_weight = torch.empty((1280, out_channels), dtype = data_type)
        # self.time_emb_proj_bias = torch.empty((out_channels), dtype = data_type)
        self.time_emb_proj = QLinear(1280, out_channels, data_type=data_type)
        self.norm2_gamma = torch.empty((out_channels), dtype = data_type)
        self.norm2_beta = torch.empty((out_channels), dtype = data_type)
        self.conv2_weight = torch.empty((out_channels, out_channels, 3, 3), dtype = data_type)
        self.conv2_bias = torch.empty((out_channels), dtype = data_type)
        self.conv_shortcut = conv_shortcut
        if self.conv_shortcut:
            self.convshortcut_weight = torch.empty((out_channels, in_channels, 1, 1), dtype = data_type)
            self.convshortcut_bias = torch.empty((out_channels), dtype = data_type)

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = F.group_norm(hidden_states, 32, self.norm1_gamma, self.norm1_beta,1e-05)
        hidden_states = F.silu(hidden_states)

        hidden_states = F.conv2d(hidden_states, self.conv1_weight, self.conv1_bias, 1, 1)

        temb = F.silu(temb)
        #temb = (torch.matmul(temb, self.time_emb_proj_weight)+self.time_emb_proj_bias)[:, :, None, None]
        #temb = (torch.addmm(self.time_emb_proj_bias, temb, self.time_emb_proj_weight))[:, :, None, None]
        temb = self.time_emb_proj.forward(temb)[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = F.group_norm(hidden_states, 32, self.norm2_gamma, self.norm2_beta,1e-05)

        hidden_states = F.silu(hidden_states)

        hidden_states = F.conv2d(hidden_states, self.conv2_weight, self.conv2_bias, 1, 1)

        if self.conv_shortcut:
            input_tensor = F.conv2d(input_tensor, self.convshortcut_weight, self.convshortcut_bias, 1)

        output_tensor = input_tensor + hidden_states

        return output_tensor