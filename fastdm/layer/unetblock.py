# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py

import torch
import torch.nn.functional as F

from fastdm.layer.resnetblock import ResnetBlock2D
from fastdm.layer.qlinear import QLinear
from fastdm.kernel.operators_set import scaled_dot_product_attention, gelu_and_mul

class Attention_SDXL:
    def __init__(self, inner_dim, cross_attention_dim=None, num_heads=None, has_ipadapter = False, ipadp_scale = 0.6, data_type = torch.float16):
        super(Attention_SDXL, self).__init__()

        self.inner_dim = inner_dim

        if num_heads is None:
            self.head_dim = 64
            self.num_heads = inner_dim // self.head_dim
        else:
            self.num_heads = num_heads
            self.head_dim = inner_dim // num_heads

        self.scale = self.head_dim**-0.5
        self.has_ipadapter = has_ipadapter
        self.ipadp_scale = ipadp_scale if has_ipadapter else None

        # if cross_attention_dim is None:
        #     cross_attention_dim = inner_dim

        if cross_attention_dim is None: #attn1
            self.qkv_proj = QLinear(inner_dim, inner_dim*3, bias=False, data_type=data_type)
        else: #attn2
            self.q_proj = QLinear(inner_dim, inner_dim, bias=False, data_type=data_type)
            self.kv_proj = QLinear(cross_attention_dim, inner_dim*2, bias=False, data_type=data_type)

        if self.has_ipadapter:
            self.ipadp_kv_proj = QLinear(cross_attention_dim, inner_dim*2, data_type=data_type)

        self.out_proj = QLinear(inner_dim, inner_dim, data_type=data_type)

    def forward(self, hidden_states, encoder_hidden_states=None, batch=2, extra_options= {}):

        # if self.qkv_weight is not None:
        if hasattr(self, "qkv_proj"):
            qkv_out = self.qkv_proj.forward(hidden_states)
            q = qkv_out[:,0:self.inner_dim].view(batch, -1, self.inner_dim)
            k = qkv_out[:,self.inner_dim:(2*self.inner_dim)].view(batch, -1, self.inner_dim)
            v = qkv_out[:,(2*self.inner_dim):(3*self.inner_dim)].view(batch, -1, self.inner_dim)

        else: #attn2
            if self.has_ipadapter and isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states, ip_neg_hidden_states = encoder_hidden_states

            q = self.q_proj.forward(hidden_states).view(batch, -1, self.inner_dim)
            kv_out = (
                    self.kv_proj.forward(encoder_hidden_states.view(encoder_hidden_states.shape[0]*encoder_hidden_states.shape[1], encoder_hidden_states.shape[2]))
                    if encoder_hidden_states is not None
                    else self.kv_proj.forward(hidden_states)
            )
            k = kv_out[:,0:self.inner_dim].view(batch, -1, self.inner_dim)
            v = kv_out[:,self.inner_dim:(2*self.inner_dim)].view(batch, -1, self.inner_dim)
        
            # comfyui ipadapter patch kwargs, if it is None, current block not compute ipadapter
            ipadapter_kwargs = extra_options.get("ipadapter_kwargs", None)

            if self.has_ipadapter:
                # for diffusers and comfyui ipadapter, there is 3 cases:
                # 1) diffusers: ipadapter_kwargs is None and extra_options is empty
                # 2) comfyui compute ipadapter: ipadapter_kwargs is not None and sigmas is in [sigmas_start, sigmas_end]
                # 3) comfyui not compute ipadapter: ipadapter_kwargs is None and extra_options is not empty
                def compute_kv(hidden_states):
                    kv_out = self.ipadp_kv_proj.forward(hidden_states[0])
                    k = kv_out[:, 0:self.inner_dim].view(batch, -1, self.inner_dim)
                    v = kv_out[:, self.inner_dim:(2 * self.inner_dim)].view(batch, -1, self.inner_dim)
                    return k, v
            
                if ipadapter_kwargs is None and not extra_options: # 1)diffusers
                    ipadp_k, ipadp_v = compute_kv(ip_hidden_states)
                    if ip_neg_hidden_states is not None:
                        ipadp_neg_k, ipadp_neg_v = compute_kv(ip_neg_hidden_states)

                        ip_k = torch.cat([ipadp_neg_k, ipadp_k], dim=0).view(batch, -1, self.inner_dim)
                        ip_v = torch.cat([ipadp_neg_v, ipadp_v], dim=0).view(batch, -1, self.inner_dim)
                    else:
                        ip_k = ipadp_k
                        ip_v = ipadp_v
                elif ipadapter_kwargs is not None: # 2) comfyui compute ipadapter
                    sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9
                    sigma_right = sigma <= ipadapter_kwargs[0]["sigma_start"] and sigma >= ipadapter_kwargs[0]["sigma_end"]
                    if sigma_right:
                        # for comfyui ipadapter, update ipadp scale and get ip_hidden_states, ip_neg_hidden_states from extra_options
                        self.ipadp_scale,ip_hidden_states,ip_neg_hidden_states = self.update_scale_and_conuncon(q, 
                                                                                                                ipadapter_kwargs, 
                                                                                                                extra_options["cond_or_uncond"], 
                                                                                                                extra_options["block"][0], 
                                                                                                                extra_options["transformer_index"])

                        ipadp_k, ipadp_v = compute_kv(ip_hidden_states)
                        if ip_neg_hidden_states is not None:
                            ipadp_neg_k, ipadp_neg_v = compute_kv(ip_neg_hidden_states)

                            ip_k = torch.cat([ipadp_neg_k, ipadp_k], dim=0).view(batch, -1, self.inner_dim)
                            ip_v = torch.cat([ipadp_neg_v, ipadp_v], dim=0).view(batch, -1, self.inner_dim)
                        else:
                            ip_k = ipadp_k
                            ip_v = ipadp_v
                        
                        # for comfyui ipadapter, update k v according to embeds_scaling, support['K+mean(V) w/ C penalty','K+V w/ C penalty','K+V' ]
                        embeds_scaling, ip_k, ip_v = self.update_ipkv(self.ipadp_scale, ip_k, ip_v, extra_options)

        attn_output = scaled_dot_product_attention(q, k, v, self.num_heads, self.num_heads, self.head_dim, scale=self.scale)

        if self.has_ipadapter:
            # there are 3 cases same as above
            tmp_attn_out = scaled_dot_product_attention(q, ip_k, ip_v, self.num_heads, self.num_heads, self.head_dim, scale=self.scale)
            if ipadapter_kwargs is None and not extra_options:
                attn_output = attn_output + self.ipadp_scale * tmp_attn_out
            elif ipadapter_kwargs is not None:
                if sigma_right:
                    ip_attn_output = self.ipadp_scale * tmp_attn_out if embeds_scaling == "V only" else tmp_attn_out
                    attn_output = attn_output + ip_attn_output
        attn_output = attn_output.view(-1, self.inner_dim)
        attn_output = self.out_proj.forward(attn_output)

        return attn_output
    
    def update_scale_and_conuncon(self, q, ipadapter_kwargs,cond_or_uncond, block_type, t_idx):
        # update ipadapter scale according node weight
        cond_alt = None
        ipadapter = ipadapter_kwargs[0]["ipadapter"]
        weight = ipadapter_kwargs[0]["weight"]
        weight_type = ipadapter_kwargs[0]["weight_type"]
        cond = ipadapter_kwargs[0]["cond"]
        uncond = ipadapter_kwargs[0]["uncond"]

        layers = 11 if '101_to_k_ip' in ipadapter.ip_layers.to_kvs else 16
        b = q.shape[0]

        if weight_type == 'ease in':
            weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
        elif weight_type == 'ease out':
            weight = weight * (0.05 + 0.95 * (t_idx / layers))
        elif weight_type == 'ease in-out':
            weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers/2)) / (layers/2)))
        elif weight_type == 'reverse in-out':
            weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers/2)) / (layers/2)))
        elif weight_type == 'weak input' and block_type == 'input':
            weight = weight * 0.2
        elif weight_type == 'weak middle' and block_type == 'middle':
            weight = weight * 0.2
        elif weight_type == 'weak output' and block_type == 'output':
            weight = weight * 0.2
        elif weight_type == 'strong middle' and (block_type == 'input' or block_type == 'output'):
            weight = weight * 0.2
        elif isinstance(weight, dict):
            if t_idx not in weight:
                return 0, cond, uncond

            if weight_type == "style transfer precise":
                if layers == 11 and t_idx == 3:
                    uncond = cond
                    cond = cond * 0
                elif layers == 16 and (t_idx == 4 or t_idx == 5):
                    uncond = cond
                    cond = cond * 0
            elif weight_type == "composition precise":
                if layers == 11 and t_idx != 3:
                    uncond = cond
                    cond = cond * 0
                elif layers == 16 and (t_idx != 4 and t_idx != 5):
                    uncond = cond
                    cond = cond * 0

            weight = weight[t_idx]

            if cond_alt is not None and t_idx in cond_alt:
                cond = cond_alt[t_idx]
                del cond_alt
        return weight, cond, uncond

    def update_ipkv(self, weight, ip_k, ip_v, extra_options):
        # update ipadapter kv according to embeds scaling
        ipadapter_kwargs = extra_options.get("ipadapter_kwargs", None)
        embeds_scaling = ipadapter_kwargs[0]["embeds_scaling"]

        if embeds_scaling == 'K+mean(V) w/ C penalty':
            scaling = float(ip_k.shape[2]) / 1280.0
            weight = weight * scaling
            ip_k = ip_k * weight
            ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
            ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
            del ip_v_mean
        elif embeds_scaling == 'K+V w/ C penalty':
            scaling = float(ip_k.shape[2]) / 1280.0
            weight = weight * scaling
            ip_k = ip_k * weight
            ip_v = ip_v * weight
        elif embeds_scaling == 'K+V':
            ip_k = ip_k * weight
            ip_v = ip_v * weight

        return embeds_scaling, ip_k, ip_v
    
class FeedForward_SDXL:
    def __init__(self, in_features, out_features, data_type = torch.float16):
        super(FeedForward_SDXL, self).__init__()

        self.proj1 = QLinear(in_features, out_features*8, data_type=data_type)
        self.proj2 = QLinear(out_features*4, out_features, data_type=data_type)

    def forward(self, x):
        x_proj = self.proj1.forward(x)
        x1_proj = gelu_and_mul(x_proj)
        x2_proj = self.proj2.forward(x1_proj)

        return x2_proj

class Attention_IpadapterPlus(Attention_SDXL):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # for qunatization
        self.kv_weight = None
        self.kv_bias = None
        self.kv_quant_scale = None
        self.kv_input_quant_scale = None

    def forward(self, hidden_states, encoder_hidden_states = None, attention_mask = None, **cross_attention_kwargs):
        # if "image_rotary_emb" in cross_attention_kwargs:
        #     image_rotary_emb = cross_attention_kwargs["image_rotary_emb"]
        # else:
        #     image_rotary_emb = None

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = self.q_proj.forward(hidden_states)
        kv_fusion = self.kv_proj.forward(encoder_hidden_states)
        key = kv_fusion[:, :, 0:self.inner_kv_dim]
        value = kv_fusion[:, :, self.inner_kv_dim:]

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        hidden_states = scaled_dot_product_attention(query, key, value, self.heads, self.heads, head_dim, scale=self.scale)
        hidden_states = hidden_states.to(query.dtype)

        # output projection
        hidden_states = self.out_proj.forward(hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states

class BasicTransformerBlock:
    def __init__(self, hidden_size, has_ipadp=False, ipadp_scale=0.6, data_type = torch.float16):
        super(BasicTransformerBlock, self).__init__()
        self.hidden_size = hidden_size
        self.norm1_gamma = torch.empty((hidden_size), dtype = data_type)
        self.norm1_beta = torch.empty((hidden_size), dtype = data_type)
        self.attn1 = Attention_SDXL(hidden_size, data_type=data_type, has_ipadapter=False)
        self.norm2_gamma = torch.empty((hidden_size), dtype = data_type)
        self.norm2_beta = torch.empty((hidden_size), dtype = data_type)
        self.attn2 = Attention_SDXL(hidden_size, 2048, data_type=data_type, has_ipadapter=has_ipadp, ipadp_scale=ipadp_scale)
        self.norm3_gamma = torch.empty((hidden_size), dtype = data_type)
        self.norm3_beta = torch.empty((hidden_size), dtype = data_type)
        self.ff = FeedForward_SDXL(hidden_size, hidden_size, data_type=data_type)
    def forward(self, x, encoder_hidden_states=None, batch_size=2, transformer_options={}):
        residual = x

        # get extra options
        extra_options = {}
        if transformer_options:
            extra_options = self.get_extra_options(transformer_options)

        x = F.layer_norm(x, [self.hidden_size], self.norm1_gamma, self.norm1_beta, eps=1e-05)
        x = self.attn1.forward(x, batch=batch_size)
        x = x + residual

        residual = x

        x = F.layer_norm(x, [self.hidden_size], self.norm2_gamma, self.norm2_beta, eps=1e-05)
        if encoder_hidden_states is not None:
            x = self.attn2.forward(x, encoder_hidden_states, batch=batch_size, extra_options=extra_options)
        else:
            x = self.attn2.forward(x, batch=batch_size, extra_options=extra_options)
        x = x + residual

        residual = x

        x = F.layer_norm(x, [self.hidden_size], self.norm3_gamma, self.norm3_beta, eps=1e-05)
        x = self.ff.forward(x)
        x = x + residual
        return x
    
    def get_extra_options(self, transformer_options):
        extra_options = {}
        transformer_patches = {}
        transformer_patches_replace={}
        for k in transformer_options:
            if k == "patches":
                transformer_patches = transformer_options[k]
            elif k == "patches_replace":
                transformer_patches_replace = transformer_options[k]
            else:
                extra_options[k] = transformer_options[k]

        block = transformer_options.get("block", None)
        block_index = transformer_options.get("block_index", 0)

        if block is not None:
            transformer_block = (block[0], block[1], block_index)
        else:
            transformer_block = None
        
        # only for attn2
        attn2_replace_patch = transformer_patches_replace.get("attn2", {})
        block_attn2 = transformer_block
        if block_attn2 not in attn2_replace_patch:
            block_attn2 = block
        if block_attn2 in attn2_replace_patch:
            extra_options["ipadapter_kwargs"] = attn2_replace_patch[block_attn2].kwargs
        else:
            extra_options["ipadapter_kwargs"] = None
        return extra_options
    
class Transformer2DModel:
    def __init__(self, in_channels, out_channels, n_layers, has_ipadp = False, ipadp_scale = 0.6, data_type = torch.float16):
        super(Transformer2DModel, self).__init__()
        self.norm_gamma = torch.empty((in_channels), dtype = data_type)
        self.norm_beta = torch.empty((in_channels), dtype = data_type)
        self.proj_in = QLinear(in_channels, out_channels, data_type=data_type)
        self.transformer_blocks = [BasicTransformerBlock(out_channels, data_type=data_type, has_ipadp=has_ipadp, ipadp_scale=ipadp_scale) for _ in range(n_layers)]
        self.proj_out = QLinear(out_channels, out_channels, data_type=data_type)
    
    def forward(self, hidden_states, encoder_hidden_states=None,transformer_options={}):

        batch, _, height, width = hidden_states.shape
        res = hidden_states
        hidden_states = F.group_norm(hidden_states, 32, self.norm_gamma, self.norm_beta, eps=1e-06)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
            batch * height * width, inner_dim
        )

        hidden_states = self.proj_in.forward(hidden_states)

        for i, block in enumerate(self.transformer_blocks):
            if transformer_options:
                transformer_options["block_index"] = i
            hidden_states = block.forward(hidden_states, encoder_hidden_states, batch, transformer_options)
        
        # comfyui transformer model counter
        if "transformer_index" in transformer_options:
            transformer_options["transformer_index"]+=1
        hidden_states = self.proj_out.forward(hidden_states)
        hidden_states = (
            hidden_states.reshape(batch, height, width, inner_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return hidden_states + res

class Downsample2D:
    def __init__(self, in_channels, out_channels, data_type = torch.float16):
        super(Downsample2D, self).__init__()
        self.conv_weight = torch.empty((out_channels, in_channels, 3, 3), dtype = data_type)
        self.conv_bias = torch.empty((out_channels), dtype = data_type)
    def forward(self):

        return
    
class Upsample2D:
    def __init__(self):
        super(Upsample2D, self).__init__()
    def forward(self):
        return
    
class DownBlock2D:
    def __init__(self, in_channels, out_channels, data_type=torch.float16):
        super(DownBlock2D, self).__init__()
        self.resnets = [ResnetBlock2D(in_channels, out_channels, conv_shortcut=False,data_type=data_type),
                        ResnetBlock2D(out_channels, out_channels, conv_shortcut=False,data_type=data_type)]
        self.downsample_conv_weight = torch.empty((out_channels, out_channels, 3, 3), dtype = data_type)
        self.downsample_conv_bias = torch.empty((out_channels), dtype = data_type)
    def forward(self, hidden_states, temb):
        output_states = []
        for module in self.resnets:
            hidden_states = module.forward(hidden_states, temb)
            output_states.append(hidden_states)

        hidden_states = F.conv2d(hidden_states, self.downsample_conv_weight, self.downsample_conv_bias, 2, 1)
        output_states.append(hidden_states)
        
        return hidden_states, output_states

class CrossAttnDownBlock2D:
    def __init__(self, in_channels, out_channels, n_layers, has_downsamplers=True, has_ipadpt=False, ipadp_scale=0.6, data_type=torch.float16):
        super(CrossAttnDownBlock2D, self).__init__()

        self.has_cross_attention = True

        self.attentions = [
            Transformer2DModel(out_channels, out_channels, n_layers, has_ipadp=has_ipadpt, ipadp_scale=ipadp_scale, data_type=data_type),
            Transformer2DModel(out_channels, out_channels, n_layers, has_ipadp=has_ipadpt, ipadp_scale=ipadp_scale, data_type=data_type)
            ]
        self.resnets = [
            ResnetBlock2D(in_channels, out_channels, data_type=data_type),
            ResnetBlock2D(out_channels, out_channels, conv_shortcut=False, data_type=data_type)
            ]
        self.has_downsamplers = has_downsamplers
        if has_downsamplers:
            self.downsample_conv_weight = torch.empty((out_channels, out_channels, 3, 3), dtype = data_type)
            self.downsample_conv_bias = torch.empty((out_channels), dtype = data_type)
    def forward(self, hidden_states, temb, encoder_hidden_states, transformer_options={}):
        output_states = []
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            hidden_states = resnet.forward(hidden_states, temb)
            if "unet_block" in transformer_options:
                if transformer_options["unet_block"][1]==1:
                    transformer_options["block"] = (transformer_options["unet_block"][0], transformer_options["unet_block"][1] + i + 3)
                else:
                    transformer_options["block"] = (transformer_options["unet_block"][0], transformer_options["unet_block"][1] + i + 5)
            hidden_states = attn.forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                transformer_options=transformer_options,
            )
            output_states.append(hidden_states)

        if self.has_downsamplers:
            hidden_states = F.conv2d(hidden_states, self.downsample_conv_weight, self.downsample_conv_bias, 2, 1)
            # hidden_states = F.conv2d(hidden_states, self.downsample_conv_weight, self.downsample_conv_bias, (2,2), (1,1), (1,1), 1)
            output_states.append(hidden_states)
        return hidden_states, output_states
    
class CrossAttnUpBlock2D:
    def __init__(self, in_channels, out_channels, prev_output_channel, n_layers, has_ipadpt=False, ipadp_scale=0.6, data_type=torch.float16):
        super(CrossAttnUpBlock2D, self).__init__()
        self.has_cross_attention = True
        self.attentions = [
            Transformer2DModel(out_channels, out_channels, n_layers, has_ipadp=has_ipadpt, ipadp_scale=ipadp_scale, data_type=data_type),
            Transformer2DModel(out_channels, out_channels, n_layers, has_ipadp=has_ipadpt, ipadp_scale=ipadp_scale, data_type=data_type),
            Transformer2DModel(out_channels, out_channels, n_layers, has_ipadp=has_ipadpt, ipadp_scale=ipadp_scale, data_type=data_type)
            ]
        self.resnets = [
            ResnetBlock2D(prev_output_channel + out_channels, out_channels, data_type=data_type),
            ResnetBlock2D(2 * out_channels, out_channels, data_type=data_type),
            ResnetBlock2D(out_channels + in_channels, out_channels, data_type=data_type)
            ]
        self.upsample_conv_weight = torch.empty((out_channels, out_channels, 3, 3), dtype = data_type)
        self.upsample_conv_bias = torch.empty((out_channels), dtype = data_type)
    def forward(self, hidden_states, res_hidden_states_tuple, temb, encoder_hidden_states, transformer_options={}):
        for i, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet.forward(hidden_states, temb)
            if "unet_block" in transformer_options:
                transformer_options["block"] = (transformer_options["unet_block"][0], transformer_options["unet_block"][1]*3+i)
            hidden_states = attn.forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                transformer_options=transformer_options,
            )

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = F.conv2d(hidden_states, self.upsample_conv_weight, self.upsample_conv_bias, 1, 1)
        return hidden_states
    
class UpBlock2D:
    def __init__(self, in_channels, out_channels, prev_output_channel, data_type=torch.float16):
        super(UpBlock2D, self).__init__()
        self.resnets = [
                ResnetBlock2D(out_channels + prev_output_channel, out_channels, data_type=data_type),
                ResnetBlock2D(out_channels * 2, out_channels, data_type=data_type),
                ResnetBlock2D(out_channels + in_channels, out_channels, data_type=data_type),
            ]
    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet.forward(hidden_states, temb)

        return hidden_states
    
class UNetMidBlock2DCrossAttn:
    def __init__(self, in_features, has_ipadpt=False, ipadp_scale=0.6, data_type = torch.float16):
        super(UNetMidBlock2DCrossAttn, self).__init__()
        self.has_cross_attention = True
        self.attentions = [Transformer2DModel(in_features, in_features, n_layers=10, has_ipadp=has_ipadpt, ipadp_scale=ipadp_scale, data_type=data_type)]
        self.resnets = [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False, data_type=data_type),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False, data_type=data_type)
            ]
    def forward(self, hidden_states, temb=None, encoder_hidden_states=None, transformer_options={}):
        hidden_states = self.resnets[0].forward(hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if "unet_block" in transformer_options:
                transformer_options["block"] = transformer_options["unet_block"]
            hidden_states = attn.forward(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                transformer_options=transformer_options,
            )
            hidden_states = resnet.forward(hidden_states, temb)

        return hidden_states