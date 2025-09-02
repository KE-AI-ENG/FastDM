
import comfy.model_patcher
import folder_paths
import GPUtil
import torch
from comfy.supported_models import SDXL, Flux, SD3, QwenImage
from comfy.controlnet import ControlNet


from fastdm.model.sdxl import SDXLUNetModelCore
from fastdm.model.controlnets import SdxlControlNetModelCore, FluxControlNetModelCore
from fastdm.model.flux import FluxTransformer2DModelCore
from fastdm.model.sd35 import SD3TransformerModelCore
from fastdm.model.qwenimage import QwenImageTransformer2DModelCore
from fastdm.comfyui_entry import (
    ComfyUIUNetForwardWrapper,
    ComfyUIControlnetForwardWrapper,
    ComfyUIFluxForwardWrapper,
    ComfyUIFluxControlnetForwardWrapper,
    ComfyUISD35ForwardWrapper,
    ComfyUIQwenImageForwardWrapper
)
from fastdm.kernel.utils import set_global_backend
from fastdm.cache_config import CacheConfig

QUANT_DTYPE_MAP = {
    "int8": torch.int8,
    "fp8": torch.float8_e4m3fn,
    "none": None,
}

class FastdmSDXLUnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_model_name": (folder_paths.get_filename_list("unet"),),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": len(GPUtil.getGPUs()), "step": 1, "display": "number", "lazy": True},
                ),
                "quant_dtype": (["fp8", "int8", "none"], {"default": "fp8"}),
                "kernel_backend": (["torch", "cuda", "triton"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Fastdm"
    TITLE = "Fastdm SDXL UNet Loader"

    def load_model(self, unet_model_name: str,device_id: int, quant_dtype: str, kernel_backend: str, **kwargs):
        # Set the global backend for FastDM
        set_global_backend(kernel_backend)

        device = f"cuda:{device_id}"
        ckpt_path = folder_paths.get_full_path_or_raise("unet", unet_model_name)
        fastdm_core_model = SDXLUNetModelCore(data_type=torch.float16, quant_dtype=QUANT_DTYPE_MAP[quant_dtype])
        fastdm_core_model.weight_loading(ckpt_path)

        unet_config = {
            "image_size": 32, 
            "use_spatial_transformer": True, 
            "legacy": False, 
            "num_classes": "sequential", 
            "adm_in_channels": 2816, 
            "in_channels": 4, 
            "out_channels": 4, 
            "model_channels": 320, 
            "num_res_blocks": [2, 2, 2], 
            "transformer_depth": [0, 0, 2, 2, 10, 10], 
            "transformer_depth_output": [0, 0, 0, 2, 2, 2, 10, 10, 10], 
            "channel_mult": [1, 2, 4], 
            "transformer_depth_middle": 10, 
            "use_linear_in_transformer": True, 
            "context_dim": 2048, 
            "use_temporal_resblock": False, 
            "use_temporal_attention": False, 
            "num_heads": -1, 
            "num_head_channels": 64, 
            "dtype": torch.float16
        }
        comfyui_model_config = SDXL(unet_config)

        comfyui_model_config.set_inference_dtype(torch.float16, None)
        comfyui_model_config.custom_operations = None

        comfyui_model = comfyui_model_config.get_model({})
        comfyui_model.diffusion_model = ComfyUIUNetForwardWrapper(fastdm_core_model, config=unet_config)

        comfyui_model = comfy.model_patcher.ModelPatcher(comfyui_model, device, device_id)

        return (comfyui_model,)        
    

class FastdmSDXLContolnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet_model_name": (folder_paths.get_filename_list("controlnet"),),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": len(GPUtil.getGPUs()), "step": 1, "display": "number", "lazy": True},
                ),
                "quant_dtype": (["fp8", "int8", "none"], {"default": "fp8"}),
                "kernel_backend": (["torch", "cuda", "triton"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_model"
    CATEGORY = "Fastdm"
    TITLE = "Fastdm SDXL Controlnet Loader"

    def load_model(self, controlnet_model_name: str, device_id: int, quant_dtype: str, kernel_backend: str, **kwargs):
        # Set the global backend for FastDM
        set_global_backend(kernel_backend)

        device = f"cuda:{device_id}"
        ckpt_path = folder_paths.get_full_path_or_raise("controlnet", controlnet_model_name)
        fastdm_control_model = SdxlControlNetModelCore(data_type=torch.float16, quant_dtype=QUANT_DTYPE_MAP[quant_dtype])
        fastdm_control_model.weight_loading(ckpt_path)

        comfyui_control_model = ComfyUIControlnetForwardWrapper(fastdm_control_model, config=None)

        controlnet = ControlNet(control_model=comfyui_control_model, global_average_pooling=False, load_device=device, manual_cast_dtype=None)
        return (controlnet,)


class FastdmFluxLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_model_name": (folder_paths.get_filename_list("unet"),),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": len(GPUtil.getGPUs()), "step": 1, "display": "number", "lazy": True},
                ),
                "quant_dtype": (["fp8", "int8", "none"], {"default": "fp8"}),
                "kernel_backend": (["torch", "cuda", "triton"], {"default": "cuda"}),
                "in_channels": ([128, 64],),
                "use_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Fastdm"
    TITLE = "Fastdm Flux Transfoemer Loader"

    def load_model(self, 
                   unet_model_name: str,
                   device_id: int,
                   quant_dtype: str,
                   kernel_backend: str,
                   in_channels: int, 
                   use_cache: bool,
                   cache_threshold: float,
                   **kwargs):
        # Set the global backend for FastDM
        set_global_backend(kernel_backend)

        device = f"cuda:{device_id}"
        ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_model_name)

        # cache config
        cache_config = CacheConfig(
            enable_caching=use_cache,
            threshold=cache_threshold,
            coefficients=[4.98651651e+02, -2.83781631e+02,  5.58554382e+01, -3.82021401e+00, 2.64230861e-01],
        )

        fastdm_core_model = FluxTransformer2DModelCore(in_channels=in_channels, 
                                                out_channels=64, 
                                                guidance_embeds=True, 
                                                data_type=torch.bfloat16, 
                                                quant_dtype=QUANT_DTYPE_MAP[quant_dtype],
                                                cache_config=cache_config,)
        
        # load model weights
        fastdm_core_model.weight_loading(ckpt_path)

        unet_config = {
            "image_model": "flux",
            "dtype": torch.bfloat16,
            "in_channels": in_channels,
            "patch_size": 1,
            "out_channels": 64,
            "vec_in_dim": 768,
            "context_in_dim": 4096,
            "hidden_size": 3072,
            "mlp_ratio": 4.0,
            "num_heads": 24,
            "depth": 19,
            "depth_single_blocks": 38,
            "axes_dim": [16, 56, 56],
            "theta": 10000,
            "qkv_bias": True,
            "guidance_embed": True,
        }
        comfyui_model_config = Flux(unet_config)

        comfyui_model_config.set_inference_dtype(torch.float16, None)
        comfyui_model_config.custom_operations = None

        comfyui_model = comfyui_model_config.get_model({})
        comfyui_model.diffusion_model = ComfyUIFluxForwardWrapper(fastdm_core_model, config=unet_config)

        comfyui_model = comfy.model_patcher.ModelPatcher(comfyui_model, device, device_id)

        return (comfyui_model,)  

class FastdmFLuxControlnetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "controlnet_model_name": (folder_paths.get_filename_list("controlnet"),),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": len(GPUtil.getGPUs()), "step": 1, "display": "number", "lazy": True},
                ),
                "quant_dtype": (["fp8", "int8", "none"], {"default": "fp8"}),
                "kernel_backend": (["torch", "cuda", "triton"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_model"
    CATEGORY = "Fastdm"
    TITLE = "Fastdm Flux Controlnet Loader"

    def load_model(
            self, 
            controlnet_model_name: str, 
            device_id: int, 
            quant_dtype: str,
            kernel_backend: str,
        ):
        # Set the global backend for FastDM
        set_global_backend(kernel_backend)

        device = f"cuda:{device_id}"
        ckpt_path = folder_paths.get_full_path_or_raise("controlnet", controlnet_model_name)
        fastdm_control_model = FluxControlNetModelCore(data_type=torch.bfloat16, quant_dtype=QUANT_DTYPE_MAP[quant_dtype], guidance_embeds=True)
        fastdm_control_model.weight_loading(ckpt_path)
        fastdm_control_model = ComfyUIFluxControlnetForwardWrapper(fastdm_control_model, config=None)

        latent_format = comfy.latent_formats.Flux()
        extra_conds = ['y', 'guidance']
        controlnet = ControlNet(fastdm_control_model, compression_ratio=1, latent_format=latent_format, concat_mask=False, load_device=device, manual_cast_dtype=None, extra_conds=extra_conds)
        return (controlnet,)

class FastdmSD35Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_model_name": (folder_paths.get_filename_list("unet"),),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": len(GPUtil.getGPUs()), "step": 1, "display": "number", "lazy": True},
                ),
                "quant_dtype": (["fp8", "int8", "none"], {"default": "fp8"}),
                "kernel_backend": (["torch", "cuda", "triton"], {"default": "cuda"}),
                "use_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Fastdm"
    TITLE = "Fastdm Flux Transfoemer Loader"

    def load_model(self, 
                   unet_model_name: str,
                   device_id: int,
                   quant_dtype: str,
                   kernel_backend: str,
                   use_cache: bool,
                   cache_threshold: float,
                   **kwargs):
        # Set the global backend for FastDM
        set_global_backend(kernel_backend)

        device = f"cuda:{device_id}"
        ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_model_name)

        # cache config
        cache_config = CacheConfig(
            enable_caching=use_cache,
            threshold=cache_threshold,
            coefficients=[ 5.02516305e+04, -1.71350998e+04,  1.81247682e+03, -6.99267532e+01, 9.39706146e-01],
        )

        fastdm_core_model = SD3TransformerModelCore(
                                                data_type=torch.bfloat16, 
                                                quant_dtype=QUANT_DTYPE_MAP[quant_dtype],
                                                cache_config=cache_config,)
        
        # load model weights
        fastdm_core_model.weight_loading(ckpt_path)

        unet_config = {
            "in_channels": 16,
            "pos_embed_scaling_factor": None,
        }
        comfyui_model_config = SD3(unet_config)

        comfyui_model_config.set_inference_dtype(torch.float16, None)
        comfyui_model_config.custom_operations = None

        comfyui_model = comfyui_model_config.get_model({})
        comfyui_model.diffusion_model = ComfyUISD35ForwardWrapper(fastdm_core_model, config=unet_config)

        comfyui_model = comfy.model_patcher.ModelPatcher(comfyui_model, device, device_id)

        return (comfyui_model,)  

class FastdmQwenImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_model_name": (folder_paths.get_filename_list("unet"),),
                "device_id": (
                    "INT",
                    {"default": 0, "min": 0, "max": len(GPUtil.getGPUs()), "step": 1, "display": "number", "lazy": True},
                ),
                "quant_dtype": (["fp8", "int8", "none"], {"default": "fp8"}),
                "kernel_backend": (["torch", "cuda", "triton"], {"default": "cuda"}),
                "use_cache": ("BOOLEAN", {"default": False}),
                "cache_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Fastdm"
    TITLE = "Fastdm QwenImage Transfoemer Loader"

    def load_model(self, 
                    unet_model_name: str,
                    device_id: int,
                    quant_dtype: str,
                    kernel_backend: str,
                    use_cache: bool,
                    cache_threshold: float,
                    **kwargs):
        # Set the global backend for FastDM
        set_global_backend(kernel_backend)

        device = f"cuda:{device_id}"
        ckpt_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_model_name)

        # cache config
        cache_config = CacheConfig(
            enable_caching=use_cache,
            threshold=cache_threshold,
            coefficients=[20.04634615, 3.13881129, -11.25528647, 4.70808005, -0.15457715],
            negtive_cache=True,
            negtive_coefficients=[ -0.23545113,24.80886833, -19.46151587, 5.9431741, -0.20358595]
        )
        fastdm_core_model = QwenImageTransformer2DModelCore(
                                                data_type=torch.bfloat16, 
                                                quant_dtype=QUANT_DTYPE_MAP[quant_dtype],
                                                cache_config=cache_config,)
        
        # load model weights
        fastdm_core_model.weight_loading(ckpt_path)

        unet_config = {
            "image_model": "qwen_image",
        }
        comfyui_model_config = QwenImage(unet_config)

        comfyui_model_config.set_inference_dtype(torch.float16, None)
        comfyui_model_config.custom_operations = None

        comfyui_model = comfyui_model_config.get_model({})
        comfyui_model.diffusion_model = ComfyUIQwenImageForwardWrapper(fastdm_core_model, config=unet_config)

        comfyui_model = comfy.model_patcher.ModelPatcher(comfyui_model, device, device_id)

        return (comfyui_model,)  
    
NODE_CLASS_MAPPINGS = {
    "FastdmSDXLUnetLoader": FastdmSDXLUnetLoader,
    "FastdmSDXLContolnetLoader": FastdmSDXLContolnetLoader,
    "FastdmFluxLoader": FastdmFluxLoader,
    "FastdmFLuxControlnetLoader": FastdmFLuxControlnetLoader,
    "FastdmSD35Loader":FastdmSD35Loader,
    "FastdmQwenImageLoader":FastdmQwenImageLoader,
}
