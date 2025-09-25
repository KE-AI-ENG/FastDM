"""
Diffusers pipeline model entry
"""

import os
import json
import types
import gc
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np

import torch
import torch.nn as nn

from diffusers import DiffusionPipeline, WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import load_image

from fastdm.model.sdxl import SDXLUNetModelCore
from fastdm.model.sd35 import SD3TransformerModelCore
from fastdm.model.flux import FluxTransformer2DModelCore
from fastdm.model.qwenimage import QwenImageTransformer2DModelCore
from fastdm.model.wan import WanTransformer3DModelCore
from fastdm.model.controlnets import SdxlControlNetModelCore, FluxControlNetModelCore
from fastdm.kernel.utils import set_global_backend
from fastdm.caching.xcaching import AutoCache
from fastdm.sparse.xsparse import RadialAttn

class ConfigMixin:
    """Configuration-related mixin class"""
    
    def _create_diffusers_config(self, **config_params):
        """Create config object for diffusers compatibility"""
        config = types.SimpleNamespace(**config_params)
        return config


class BaseModelWrapper(nn.Module, ConfigMixin):
    """base model wrapper class"""
    
    def __init__(self, kernel_backend: str = "torch", config_path: Optional[str] = None):
        super().__init__()
        self._setup_backend(kernel_backend)
        self.model_config_dict = self._load_config(config_path)
        self.device = self._get_device()
        self.core_model = None
    
    def _setup_backend(self, kernel_backend: str):
        """Setup computation backend"""
        try:
            set_global_backend(kernel_backend)
        except Exception as e:
            print(f"Warning: Failed to set backend {kernel_backend}: {e}")
    
    def _get_device(self) -> torch.device:
        """Get computation device"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_config(self, config_path: Optional[str]) -> Optional[Dict]:
        """Load configuration file"""
        if not config_path or not os.path.isfile(config_path):
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return None
    
    @contextmanager
    def cache_context(self, name: str):
        r"""For diffusers-v0.35 compatibility, but no functionality is provided. Context manager that provides additional methods for cache management."""
        # print(f"Entering cache context: {name}")
        try:
            yield
        except Exception as e:
            print(f"Exception in cache context '{name}': {type(e).__name__}: {e}")
            raise
        finally:
            # print(f"Exiting cache context: {name}")
            pass
    
    def to(self, *args, **kwargs):
        """Overrides the default `to` method to ensure that all submodules are moved to the correct device and dtype."""
        return
    
    def forward(self, *args, **kwargs):
        """Base class forward method"""
        if self.core_model is None:
            raise NotImplementedError("Core model not initialized")
        return self.core_model.forward(*args, **kwargs)


class FluxTransformerWrapper(BaseModelWrapper):
    """Flux Transformer wrapper"""
    
    def __init__(
        self, 
        ckpt_path = None,
        in_channels: int = 64, 
        out_channels: int = 64,
        dtype: torch.dtype = torch.bfloat16, 
        quant_type: Optional[torch.dtype] = None, 
        kernel_backend: str = "torch",
        need_resolve_oom=False,
        cache: AutoCache = None,
        **kwargs
    ):
        super().__init__(kernel_backend)
        
        # Create configuration
        self.config = self._create_diffusers_config(
            in_channels=in_channels,
            out_channels=out_channels,
            guidance_embeds=True,
            dtype=dtype
        )
        self.need_resolve_oom = need_resolve_oom
        self.dtype = dtype
        self._initialize_core_model(ckpt_path, quant_type, cache, **kwargs)
    
    def _initialize_core_model(self, ckpt_path, quant_type, cache, **kwargs):
        """Initialize core model"""
        self.core_model = FluxTransformer2DModelCore(
            in_channels=self.config.in_channels, 
            out_channels=self.config.out_channels,
            guidance_embeds=self.config.guidance_embeds, 
            data_type=self.config.dtype,
            quant_dtype=quant_type,
            cache=cache,
            oom_ressolve=self.need_resolve_oom,
            **kwargs
        )
        
        if isinstance(ckpt_path, dict) or os.path.exists(ckpt_path):
            self.core_model.weight_loading(ckpt_path)
        elif ckpt_path:
            print(f"Warning: Checkpoint path {ckpt_path} does not exist")

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Forward pass"""
        return self.core_model.forward(hidden_states, **kwargs)


class QwenTransformerWrapper(BaseModelWrapper):
    """Qwen Transformer wrapper"""
    
    def __init__(
        self, 
        ckpt_path = None,
        in_channels: int = 64, 
        out_channels: int = 64,
        dtype: torch.dtype = torch.bfloat16, 
        quant_type: Optional[torch.dtype] = None, 
        kernel_backend: str = "torch",
        need_resolve_oom=False,
        cache = None,
        **kwargs
    ):
        super().__init__(kernel_backend)
        
        self.config = self._create_diffusers_config(
            in_channels=in_channels,
            out_channels=out_channels,
            guidance_embeds=False,
            dtype=dtype
        )
        
        self.dtype = dtype

        self.need_resolve_oom = need_resolve_oom

        self._initialize_core_model(ckpt_path, quant_type, cache, **kwargs)
    
    def _initialize_core_model(self, ckpt_path, quant_type, cache, **kwargs):
        """Initialize core model"""
        self.core_model = QwenImageTransformer2DModelCore(
            in_channels=self.config.in_channels, 
            out_channels=self.config.out_channels,
            guidance_embeds=self.config.guidance_embeds, 
            data_type=self.config.dtype,
            quant_dtype=quant_type,
            cache=cache,
            oom_ressolve=self.need_resolve_oom,
            **kwargs
        )
        
        if isinstance(ckpt_path, dict) or os.path.exists(ckpt_path):
            self.core_model.weight_loading(ckpt_path)


class SD35TransformerWrapper(BaseModelWrapper):
    """SD3.5 Transformer wrapper"""
    
    def __init__(
        self, 
        ckpt_path = None,
        in_channels: int = 16, 
        out_channels: int = 16,
        dtype: torch.dtype = torch.bfloat16, 
        quant_type: Optional[torch.dtype] = None, 
        kernel_backend: str = "torch",
        need_resolve_oom=False,
        cache = None,
        **kwargs
    ):
        super().__init__(kernel_backend)
        
        self.config = self._create_diffusers_config(
            in_channels=in_channels,
            out_channels=out_channels,
            joint_attention_dim=4096,
            sample_size=128,
            dtype=dtype
        )
        
        self.dtype = dtype
        self._initialize_core_model(ckpt_path, quant_type, cache, **kwargs)
    
    def _initialize_core_model(self, ckpt_path, quant_type, cache, **kwargs):
        """Initialize core model"""
        self.core_model = SD3TransformerModelCore(
            in_channels=self.config.in_channels, 
            out_channels=self.config.out_channels,
            joint_attention_dim=self.config.joint_attention_dim,
            sample_size=self.config.sample_size,
            data_type=self.config.dtype,
            quant_dtype=quant_type,
            cache=cache,
            **kwargs
        )
        
        if isinstance(ckpt_path, dict) or os.path.exists(ckpt_path):
            self.core_model.weight_loading(ckpt_path)


class SDXLUNetModelWrapper(BaseModelWrapper):
    """SDXL UNet model wrapper"""
    
    def __init__(
        self, 
        ckpt_path=None,
        in_channels=4, 
        dtype=torch.bfloat16, 
        is_ipadapter=False, 
        ipadapter_scale=0.6, 
        diffu_ipadapter_encoder_hid_proj = None, 
        quant_type=None, 
        kernel_backend="torch",
        **kwargs
    ):
        super().__init__(kernel_backend)
        
        self.config = self._create_diffusers_config(
            in_channels=in_channels,
            addition_time_embed_dim = 256,
            sample_size = 128,
            time_cond_proj_dim = None,
            ip_adapter = is_ipadapter,
            dtype = dtype
        )
        
        self.dtype = dtype
        self._initialize_core_model(ckpt_path, quant_type, ipadapter_scale, diffu_ipadapter_encoder_hid_proj, **kwargs)
    
    def _initialize_core_model(self, ckpt_path=None, quant_type=None, ipadapter_scale=0.6, diffu_ipadapter_encoder_hid_proj = None, **kwargs):
        """Initialize core model"""
        self.core_model = SDXLUNetModelCore(
            in_channels=self.config.in_channels,
            is_ip_adapter=self.config.ip_adapter, 
            ip_adapter_scale=ipadapter_scale,
            diffu_ipadapter_encoder_hid_proj = diffu_ipadapter_encoder_hid_proj, 
            data_type=self.config.dtype,
            quant_dtype=quant_type,
            **kwargs
        )
        
        self.add_embedding = self.core_model.add_embedding
        self.encoder_hid_proj = self.core_model.encoder_hid_proj

        if isinstance(ckpt_path, dict) or os.path.exists(ckpt_path):
            self.core_model.weight_loading(ckpt_path)

    @torch.no_grad()
    def forward(
        self, sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs
    ):
        return self.core_model.forward(sample, timesteps, encoder_hidden_states, added_cond_kwargs, **kwargs)


class SDXLControlnetModelWrapper(BaseModelWrapper):
    """SDXL ControlNet model wrapper"""
    
    def __init__(
        self, 
        ckpt_config = None,
        ckpt_path = None,
        in_channels: int = 4, 
        dtype: torch.dtype = torch.float16, 
        quant_type: Optional[torch.dtype] = None, 
        kernel_backend: str = "torch",
        **kwargs
    ):
        super().__init__(kernel_backend)
        
        self.config = self._create_diffusers_config(
            global_pool_conditions = False
        )
        
        self.dtype = dtype
        self._initialize_core_model(ckpt_config, ckpt_path, quant_type, **kwargs)
    
    def _initialize_core_model(self, ckpt_config=None, ckpt_path=None, quant_type=None, **kwargs):
        """Initialize core model"""
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
            self.core_model = SdxlControlNetModelCore(**filtered_config, quant_dtype=quant_type)
        else:
            self.core_model = SdxlControlNetModelCore(data_type=self.dtype, quant_dtype=quant_type)
        
        if isinstance(ckpt_path, dict) or os.path.exists(ckpt_path):
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

class FluxControlnetWrapper(BaseModelWrapper):
    """Flux ControlNet wrapper"""
    
    def __init__(
        self, 
        ckpt_path: Optional[str] = None,
        in_channels: int = 64, 
        out_channels: int = 64,
        dtype: torch.dtype = torch.bfloat16, 
        quant_type: Optional[torch.dtype] = None, 
        kernel_backend: str = "torch",
        cache = None,
        need_resolve_oom=False,
        **kwargs
    ):
        super().__init__(kernel_backend)
        
        # diffusers compatibility
        self.config = self._create_diffusers_config(
            in_channels=in_channels,
            out_channels=out_channels,
            guidance_embeds=True,
            dtype=dtype
        )
        self.input_hint_block = None
        self.need_resolve_oom = need_resolve_oom
        self.dtype = dtype
        self._initialize_core_model(ckpt_path, quant_type, cache, **kwargs)
    
    def _initialize_core_model(self, ckpt_path, quant_type, cache, **kwargs):
        """Initialize core model"""
        self.core_model = FluxControlNetModelCore(
            in_channels=self.config.in_channels, 
            out_channels=self.config.out_channels,
            guidance_embeds=self.config.guidance_embeds, 
            data_type=self.config.dtype,
            quant_dtype=quant_type,
            oom_ressolve=self.need_resolve_oom,
            **kwargs
        )
        
        if isinstance(ckpt_path, dict) or os.path.exists(ckpt_path):
            self.core_model.weight_loading(ckpt_path)


class WanTransformer3DWrapper(BaseModelWrapper):
    """Wan 3D Transformer wrapper"""
    
    def __init__(
        self, 
        ckpt_path: Optional[str] = None,
        in_channels: int = 16, 
        out_channels: int = 16,
        dtype: torch.dtype = torch.bfloat16, 
        quant_type: Optional[torch.dtype] = None, 
        kernel_backend: str = "torch",
        config_json=None,
        cache = None,
        **kwargs
    ):
        super().__init__(kernel_backend, config_path=config_json)
        
        self.config = self._create_diffusers_config(
            in_channels=self.model_config_dict['in_channels'],
            out_channels=self.model_config_dict['out_channels'],
            dtype=dtype,
            image_dim=self.model_config_dict['image_dim'],
            patch_size=self.model_config_dict['patch_size'],
        )
        
        self.dtype = dtype
        self._initialize_core_model(ckpt_path, quant_type, cache, **kwargs)
    
    def _initialize_core_model(self, ckpt_path, quant_type, cache, **kwargs):
        """Initialize core model"""
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
                cache=cache,
                sparse_attn=kwargs.get('sparse_attn', None),
        )
        
        if isinstance(ckpt_path, dict) or os.path.exists(ckpt_path):
            self.core_model.weight_loading(ckpt_path)


class ModelWrapperFactory:
    """Model wrapper factory class"""
    
    _WRAPPER_MAPPING = {
        'flux': FluxTransformerWrapper,
        'qwen': QwenTransformerWrapper,
        'sd35': SD35TransformerWrapper,
        'sdxl': SDXLUNetModelWrapper,
        'sdxl_controlnet': SDXLControlnetModelWrapper,
        'flux_controlnet': FluxControlnetWrapper,
        'wan': WanTransformer3DWrapper,
    }
    
    @classmethod
    def create_wrapper(cls, model_type: str, **kwargs) -> BaseModelWrapper:
        """Create model wrapper"""
        if model_type not in cls._WRAPPER_MAPPING:
            available_types = ', '.join(cls._WRAPPER_MAPPING.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Available types: {available_types}")
        
        wrapper_class = cls._WRAPPER_MAPPING[model_type]
        return wrapper_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get available model types"""
        return list(cls._WRAPPER_MAPPING.keys())
    
    @classmethod
    def register_wrapper(cls, model_type: str, wrapper_class: type):
        """Register new model wrapper"""
        if not issubclass(wrapper_class, BaseModelWrapper):
            raise TypeError("Wrapper class must inherit from BaseModelWrapper")
        cls._WRAPPER_MAPPING[model_type] = wrapper_class


# Convenience functions
def create_model(model_type: str, **kwargs) -> BaseModelWrapper:
    """Convenience function to create model"""
    return ModelWrapperFactory.create_wrapper(model_type, **kwargs)


def list_available_models() -> List[str]:
    """List all available model types"""
    return ModelWrapperFactory.get_available_models()


# # Usage examples
# if __name__ == "__main__":
#     # Create Flux model
#     flux_model = create_model(
#         'flux',
#         ckpt_path='path/to/flux/checkpoint.pth',
#         dtype=torch.bfloat16,
#         kernel_backend='torch'
#     )
    
#     # Use cache context for inference
#     with flux_model.cache_context("inference"):
#         # Mock input
#         hidden_states = torch.randn(1, 64, 32, 32, dtype=torch.bfloat16)
#         output = flux_model(hidden_states)
    
#     # List all available models
#     print("Available models:", list_available_models())

class FastDMEngine:
    """FastDM Inference Engine"""
    
    def __init__(self,
                 model_path,
                 architecture="sdxl",
                 device=0,
                 data_type="bfloat16",
                 use_fp8=False,
                 use_int8=False,
                 kernel_backend="cuda",
                 cache_config=None,
                 oom_resolve=False,
                 use_diffusers = True,
                 task="t2i",
                 enable_sparse_attn=False):
        """
        初始化 FastDM 引擎
        
        Args:
            model_path: 模型路径
            architecture: 模型架构 (sdxl/flux/sd3/qwen/wan)
            device: GPU设备号
            data_type: 数据类型 (float16/bfloat16)
            use_fp8: 是否使用 FP8
            use_int8: 是否使用 INT8
            kernel_backend: 后端类型 (cuda/triton/torch)
            cache_config: 缓存配置文件路径
            oom_resolve: 是否启用 OOM 解决方案
            use_diffusers: 是否使用 diffusers 库
            task: 任务类型 (t2i/t2v/i2i/i2v)
            enable_sparse_attn: 是否启用稀疏注意力机制
        """
        self.architecture = architecture
        self.device = device
        self.oom_resolve = oom_resolve
        self.use_diffusers = use_diffusers
        self.task = task
        self.enable_sparse_attn = enable_sparse_attn
        
        # 设置设备
        torch.cuda.set_device(device)
        
        # 设置数据类型
        self.dtype = torch.bfloat16 if data_type == "bfloat16" else torch.float16
        
        # 设置量化类型
        if use_fp8:
            self.quant_type = torch.float8_e4m3fn
        elif use_int8:
            self.quant_type = torch.int8
        else:
            self.quant_type = None
            
        # 初始化caching
        if cache_config:
            self.cache = AutoCache.from_json(cache_config)
            if architecture == "wan" or architecture == "wan-i2v":
                self.cache_2 = AutoCache.from_json(cache_config)
        else:
            self.cache = None
            self.cache_2 = None

        # 初始化sparse attention
        if self.enable_sparse_attn:
            if architecture not in ["wan", "wan-i2v"]:
                raise ValueError("Sparse attention is only supported for Wan models")
            self.sparse_attn = RadialAttn(block_size=64,model_type="wan")
            
        # 初始化模型
        self._init_model(model_path, kernel_backend)
        
    def _init_model(self, model_path, kernel_backend):
        """初始化模型"""
        if self.architecture == "wan":
            vae = AutoencoderKLWan.from_pretrained(
                model_path, 
                subfolder="vae",
                torch_dtype=torch.float32
            )
            self.pipe = WanPipeline.from_pretrained(
                model_path,
                vae=vae,
                torch_dtype=torch.bfloat16
            )
        elif self.architecture == "wan-i2v":
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                use_safetensors=True
            )
        
        if self.use_diffusers:
            self.pipe.to("cuda")
        else:
            self.model_path = model_path

            # 设置caching回调
            if self.cache:
                self.cache.config.current_steps_callback = lambda: self.pipe.scheduler.step_index
                self.cache.config.total_steps_callback = lambda: self.pipe.scheduler.timesteps.shape[0]
                if self.architecture == "wan" or self.architecture == "wan-i2v": # wan model use diff cache for low noise and high noise
                    self.cache_2.config.current_steps_callback = lambda: self.pipe.scheduler.step_index
                    self.cache_2.config.total_steps_callback = lambda: self.pipe.scheduler.timesteps.shape[0]
            
            # 设置radial attn current step回调
            if self.sparse_attn:     
                self.sparse_attn.current_steps_callback = lambda: self.pipe.scheduler.step_index
                
            # 替换模型实现
            if self.architecture == "sdxl":
                self.pipe.unet = create_model(
                    "sdxl",
                    ckpt_path=self.pipe.unet.state_dict(),
                    dtype=self.dtype,
                    quant_type=self.quant_type,
                    kernel_backend=kernel_backend
                ).eval()

            elif self.architecture in ["flux", "sd35", "qwen"]:
                model_type = self.architecture

                self.pipe.transformer = create_model(
                    model_type,
                    ckpt_path=self.pipe.transformer.state_dict(),
                    dtype=self.dtype,
                    quant_type=self.quant_type,
                    kernel_backend=kernel_backend,
                    cache=self.cache,
                    need_resolve_oom=self.oom_resolve,
                ).eval()
                
                # 处理 OOM
                if self.oom_resolve and model_type in ["qwen", "flux"]:
                    self._setup_oom_resolve(model_type)

            elif "wan" == self.architecture or "wan-i2v" == self.architecture:
                self.pipe.transformer = create_model("wan", 
                                ckpt_path=self.pipe.transformer.state_dict(), 
                                dtype=self.dtype, 
                                quant_type=self.quant_type, 
                                kernel_backend=kernel_backend, 
                                config_json=f"{self.model_path}/transformer/config.json",
                                cache=self.cache,
                                sparse_attn=self.sparse_attn).eval()
                if hasattr(self.pipe, 'transformer_2') and self.pipe.transformer_2 is not None:
                    self.pipe.transformer_2 = create_model("wan", 
                                    ckpt_path=self.pipe.transformer_2.state_dict(), 
                                    dtype=self.dtype, 
                                    quant_type=self.quant_type, 
                                    kernel_backend=kernel_backend, 
                                    config_json=f"{self.model_path}/transformer_2/config.json",
                                    cache=self.cache_2,
                                    sparse_attn=self.sparse_attn).eval()
            else:
                raise ValueError(
                    f"The {self.architecture} model is not supported!!!"
                )
            
            # 设置设备
            if self.oom_resolve and self.architecture in ["qwen", "flux"]:
                self.pipe.vae.to("cuda")
            else:
                self.pipe.to("cuda")
                
            # 清理缓存
            torch.cuda.empty_cache()
            gc.collect()
        
    def _setup_oom_resolve(self, model_type):
        """设置 OOM 解决方案"""
        if model_type == "qwen":
            from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
            from fastdm.utils.qwen_vae import qwen_vae_new_decode, qwen_vae_new_encode
            AutoencoderKLQwenImage._decode = qwen_vae_new_decode
            AutoencoderKLQwenImage._encode = qwen_vae_new_encode
        else:  # flux
            from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
            from fastdm.utils.flux_vae import flux_vae_new_decode, flux_vae_new_encode
            AutoencoderKL._decode = flux_vae_new_decode
            AutoencoderKL._encode = flux_vae_new_encode
            
    def generate(self,
                prompt,
                negative_prompt=None,
                src_image=None,
                num_frames=None,
                steps=50,
                guidance_scale=3.5,
                true_cfg_scale=None,
                gen_seed=42,
                gen_width=512,
                gen_height=512,
                max_seq_len=512):
        """
        生成图像或视频
        
        Args:
            prompt: 提示词
            negative_prompt: 负面提示词
            src_image: 输入图像路径
            num_frames: 视频帧数
            fps: 视频帧率
            steps: 推理步数
            guidance_scale: 引导系数
            true_cfg_scale: QWen模型的引导系数
            gen_seed: 随机种子
            gen_width: 生成宽度
            gen_height: 生成高度
            max_seq_len: 最大序列长度
            
        Returns:
            生成的图像或视频帧
        """
        gen = torch.Generator().manual_seed(gen_seed)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "generator": gen,
            "max_sequence_length": max_seq_len
        }

        if src_image is not None and self.task == "i2v":
            processed_image, new_gen_height, new_gen_width = self.image_processor(src_image, width=gen_width, height=gen_height)
            kwargs["image"] = processed_image
        else:
            new_gen_height, new_gen_width = gen_height, gen_width

        kwargs["width"] = new_gen_width
        kwargs["height"] = new_gen_height
        
        # 处理特殊参数
        if num_frames is not None:
            kwargs["num_frames"] = num_frames
            
        if true_cfg_scale is not None:  # QWen
            kwargs["true_cfg_scale"] = true_cfg_scale
        else:
            kwargs["guidance_scale"] = guidance_scale
            
        # 生成
        with torch.inference_mode():
            output = self.pipe(**kwargs)
            
        if num_frames is not None:
            return output.frames[0]
        return output.images[0]


    def image_processor(self, input_image, width=832, height=480):
        if isinstance(input_image, str):
            if not os.path.isfile(input_image):
                raise FileNotFoundError(f"Input image file does not exist: {input_image}")
            image = load_image(input_image)
        else:
            image = input_image

        max_area = height * width
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height_ = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width_ = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        image = image.resize((width_, height_))
        return image, height_, width_