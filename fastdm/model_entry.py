import os
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import namedtuple
from contextlib import contextmanager

import torch
import torch.nn as nn

from fastdm.model.sdxl import SDXLUNetModelCore
from fastdm.model.sd35 import SD3TransformerModelCore
from fastdm.model.flux import FluxTransformer2DModelCore
from fastdm.model.qwenimage import QwenImageTransformer2DModelCore
from fastdm.model.wan import WanTransformer3DModelCore
from fastdm.model.controlnets import SdxlControlNetModelCore, FluxControlNetModelCore
from fastdm.kernel.utils import set_global_backend
from fastdm.cache_config import CacheConfig

class BaseModelWrapper(nn.Module):
    def __init__(self, kernel_backend="torch", config_path=None):
        '''
        Base class for model wrappers.
        Args:
            kernel_backend (str): The backend to use for kernel operations(torch/triton/cuda). Default is "torch".
            config_path (str, optional): Path to a JSON configuration file(in hf-checkpoints). If provided, the configuration will be loaded from this file.
        '''
        super(BaseModelWrapper, self).__init__()

        set_global_backend(kernel_backend)

        if config_path is not None and os.path.isfile(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                self.model_config_dict = json.load(f)
        else:
            self.model_config_dict = None
    
    @contextmanager
    def cache_context(self, name: str):
        r"""For diffusers-v0.35 compatibility, but no functionality is provided. Context manager that provides additional methods for cache management."""
        # print("entrance cache context, diffusers-v0.35 compatibility, but no functionality is provided.")
        try:
            yield
        except Exception as e:
            print(f"eception type: {type(e)}, eception status: {e}")
        finally:
            print("exit cache context")
    
    def to(self, *args, **kwargs):
        """Overrides the default `to` method to ensure that all submodules are moved to the correct device and dtype."""
        return


class QwenTransformerWrapper(BaseModelWrapper):
    def __init__(
            self, 
            ckpt_path=None,
            in_channels=64, 
            dtype=torch.bfloat16,
            quant_type=None, 
            kernel_backend="torch",
            cache_config: CacheConfig = None
        ):
        super().__init__(kernel_backend)

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels out_channels guidance_embeds dtype"
        )
        # flux lora will chang img_in weight shape
        self.config.in_channels = in_channels
        self.config.out_channels = 64
        self.config.guidance_embeds = False
        self.config.dtype = dtype

        self.dtype = dtype

        self.core_model = QwenImageTransformer2DModelCore(in_channels=self.config.in_channels, 
                                                        out_channels=self.config.out_channels,
                                                        guidance_embeds=self.config.guidance_embeds, 
                                                        data_type=self.config.dtype, 
                                                        quant_dtype=quant_type,
                                                        cache_config=cache_config)
        self.core_model.weight_loading(ckpt_path)

    @torch.no_grad()
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
        return self.core_model.forward(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                timestep,
                img_shapes,
                txt_seq_lens,
                guidance,  # TODO: this should probably be removed
                attention_kwargs,
                return_dict,
            )

class FluxTransformerWrapper(BaseModelWrapper):
    def __init__(
            self, 
            ckpt_path=None,
            in_channels=64, 
            dtype=torch.bfloat16, 
            quant_type=None, 
            kernel_backend="torch",
            cache_config: CacheConfig = None,):
        super().__init__(kernel_backend)

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels out_channels guidance_embeds dtype"
        )
        # flux lora will chang img_in weight shape
        self.config.in_channels = in_channels
        self.config.out_channels = 64
        self.config.guidance_embeds = True
        self.config.dtype = dtype

        self.dtype = dtype

        self.core_model = FluxTransformer2DModelCore(in_channels=self.config.in_channels, 
                                                     out_channels=self.config.out_channels,
                                                     guidance_embeds=self.config.guidance_embeds, 
                                                     data_type=self.config.dtype,
                                                     quant_dtype=quant_type,
                                                     cache_config=cache_config)
        self.core_model.weight_loading(ckpt_path)

    @torch.no_grad()
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
        controlnet_blocks_repeat: bool = False
    ):
        return self.core_model.forward(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
            joint_attention_kwargs,
            controlnet_block_samples,
            controlnet_single_block_samples,
            return_dict,
            controlnet_blocks_repeat
            )

class FluxControlnetWrapper(BaseModelWrapper):
    def __init__(
            self, 
            ckpt_path=None, 
            in_channels=64, 
            dtype=torch.bfloat16, 
            quant_type=None, 
            kernel_backend="torch"
        ):
        
        super().__init__(kernel_backend)
        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels out_channels guidance_embeds dtype"
        )
        self.config.in_channels = in_channels
        self.config.out_channels = 64
        self.config.guidance_embeds = True
        self.config.dtype = dtype
        self.input_hint_block = None

        self.dtype = dtype

        # fix error: AttributeError: 'StableDiffusionXLControlNetPipeline' object has no attribute '_execution_device'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.core_model = FluxControlNetModelCore(in_channels=self.config.in_channels, 
                                                  out_channels=self.config.out_channels,
                                                  guidance_embeds=self.config.guidance_embeds, 
                                                  data_type=self.config.dtype,
                                                  quant_dtype=quant_type
                                                  )
        self.core_model.weight_loading(ckpt_path)

    @torch.no_grad()
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
        return self.core_model.forward(
            hidden_states,
            controlnet_cond,
            controlnet_mode,
            conditioning_scale,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
            joint_attention_kwargs,
            return_dict
        )

class SD35TransformerWrapper(BaseModelWrapper):
    def __init__(
            self, 
            ckpt_path=None, 
            dtype=torch.bfloat16,
            quant_type=None,
            kernel_backend="torch",
            cache_config: CacheConfig = None
        ):
        super().__init__(kernel_backend)

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels joint_attention_dim sample_size dtype"
        )
        self.config.in_channels = 16
        self.config.joint_attention_dim = 4096
        self.config.sample_size = 128
        self.config.dtype = dtype

        self.dtype = dtype

        self.core_model = SD3TransformerModelCore(
            data_type=self.config.dtype, 
            quant_dtype=quant_type,
            cache_config=cache_config,
        )
        self.core_model.weight_loading(ckpt_path)

        return

    @torch.no_grad()
    def forward(
        self,         
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
    ):
        return self.core_model.forward(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            block_controlnet_hidden_states,
            joint_attention_kwargs
            )
    
class SDXLUNetModelWrapper(BaseModelWrapper):
    def __init__(
            self, 
            ckpt_path=None,
            in_channels=4, 
            dtype=torch.bfloat16, 
            is_ipadapter=False, 
            ipadapter_scale=0.6, 
            diffu_ipadapter_encoder_hid_proj = None, 
            quant_type=None, 
            kernel_backend="torch"):
        super().__init__(kernel_backend=kernel_backend)
        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels addition_time_embed_dim sample_size time_cond_proj_dim ip_adapter dtype"
        )
        self.config.in_channels = in_channels
        self.config.addition_time_embed_dim = 256
        self.config.sample_size = 128
        self.config.time_cond_proj_dim = None
        self.config.ip_adapter = is_ipadapter
        self.config.dtype = dtype

        self.dtype = dtype

        self.core_model = SDXLUNetModelCore(
            in_channels=in_channels,
            is_ip_adapter=is_ipadapter, 
            ip_adapter_scale=ipadapter_scale,
            diffu_ipadapter_encoder_hid_proj = diffu_ipadapter_encoder_hid_proj, 
            data_type=self.config.dtype,
            quant_dtype=quant_type,
        )

        self.add_embedding = self.core_model.add_embedding
        self.encoder_hid_proj = self.core_model.encoder_hid_proj

        self.core_model.weight_loading(ckpt_path)

    @torch.no_grad()
    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs
    ):
        return self.core_model.forward(sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs)

class SDXLControlnetModelWrapper(BaseModelWrapper):
    def __init__(self, ckpt_config=None, ckpt_path=None, dtype=torch.float16, quant_type=None, kernel_backend="torch"):
        super().__init__(kernel_backend)

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "global_pool_conditions"
        )
        self.config.global_pool_conditions = False

        self.dtype = dtype
        # fix error: AttributeError: 'StableDiffusionXLControlNetPipeline' object has no attribute '_execution_device'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def config_filtering_function(pair):
            unwanted_key = '_'
            key, value = pair
            if unwanted_key == key[0]:
                return False  # filter pair out of the dictionary
            else:
                return True  # keep pair in the filtered dictionary
        
        if ckpt_config is not None:
            print("In Debug: The config come from huggingface checkpoints!")
            filtered_config = dict(filter(config_filtering_function, ckpt_config.items()))
            self.core_model = SdxlControlNetModelCore(**filtered_config)
        else:
            self.core_model = SdxlControlNetModelCore(data_type=self.dtype)

        self.core_model.weight_loading(ckpt_path)

    @torch.no_grad()
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
        return self.core_model.forward(sample,
                                       timestep,
                                       encoder_hidden_states,
                                       controlnet_cond,
                                       conditioning_scale,
                                       class_labels,
                                       timestep_cond,
                                       attention_mask,
                                       added_cond_kwargs,
                                       cross_attention_kwargs,
                                       guess_mode,
                                       return_dict)
    

class WanTransformer3DWrapper(BaseModelWrapper):
    def __init__(
            self, 
            ckpt_path=None, 
            dtype=torch.bfloat16, 
            quant_type=None, 
            kernel_backend="torch", 
            config_json=None, 
            cache_config: CacheConfig = None):
        super().__init__(kernel_backend, config_json)

        assert self.model_config_dict is not None

        # This is needed to imitate huggingface config behavior
        # has nothing to do with the model itself
        # remove this if you don't use diffuser's pipeline
        self.config = namedtuple(
            "config", "in_channels out_channels dtype"
        )
        # flux lora will chang img_in weight shape
        self.config.in_channels = self.model_config_dict['in_channels']
        self.config.out_channels = self.model_config_dict['out_channels']
        self.config.dtype = dtype

        self.dtype = dtype

        self.core_model = WanTransformer3DModelCore(
                                                    patch_size = self.model_config_dict['patch_size'],
                                                    num_attention_heads = self.model_config_dict['num_attention_heads'],
                                                    attention_head_dim = self.model_config_dict['attention_head_dim'],
                                                    in_channels = self.model_config_dict['in_channels'],
                                                    out_channels = self.model_config_dict['out_channels'],
                                                    text_dim = self.model_config_dict['text_dim'],
                                                    freq_dim = self.model_config_dict['freq_dim'],
                                                    ffn_dim = self.model_config_dict['ffn_dim'],
                                                    num_layers = self.model_config_dict['num_layers'],
                                                    qk_norm = self.model_config_dict['qk_norm'],
                                                    eps = self.model_config_dict['eps'],
                                                    rope_max_seq_len = self.model_config_dict['rope_max_seq_len'],
                                                    data_type=self.config.dtype, 
                                                    quant_dtype=quant_type,
                                                    cache_config=cache_config)
        self.core_model.weight_loading(ckpt_path)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        return self.core_model.forward(
            hidden_states,
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            return_dict,
            attention_kwargs,
            )