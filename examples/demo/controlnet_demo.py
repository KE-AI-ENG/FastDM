import argparse
import time
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers.utils import load_image

from fastdm.model_entry import (
    FluxControlnetWrapper, FluxTransformerWrapper,
    SDXLControlnetModelWrapper, SDXLUNetModelWrapper
)
from fastdm.caching.xcaching import AutoCache

def preprocess_flux(image_path):
    """Flux: load raw image"""
    return load_image(image_path)

def preprocess_sdxl(image_path):
    """SDXL: convert canny image"""
    image = np.array(load_image(image_path))
    image = cv2.Canny(image, 100, 200)
    image = np.repeat(image[:, :, None], 3, axis=2)
    return Image.fromarray(image)

def run_pipeline(
    pipe_class,
    controlnet_class,
    transformer_wrapper=None,
    unet_wrapper=None,
    preprocess_fn=None,
    args=None
):
    # load controlnet
    controlnet = controlnet_class.from_pretrained(
        args.controlnet_model,
        torch_dtype=args.dtype
    )

    # load Pipeline
    kwargs = {}
    if args.model_type == "sdxl":
        from diffusers import AutoencoderKL
        kwargs["vae"] = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=args.dtype)

    pipe = pipe_class.from_pretrained(
        args.model_path,
        controlnet=controlnet,
        torch_dtype=args.dtype,
        **kwargs
    ).to("cuda")

    # fastdm
    if transformer_wrapper:
        if args.cache_config is not None:
            cache = AutoCache.from_json(args.cache_config)
            cache.config.current_steps_callback = lambda: pipe.scheduler.step_index
            cache.config.total_steps_callback = lambda: pipe.scheduler.timesteps.shape[0] # used by dicache
        else:
            cache = None
        pipe.transformer = transformer_wrapper(
            pipe.transformer.state_dict(),
            in_channels=64,
            quant_type=torch.float8_e4m3fn,
            dtype=args.dtype,
            kernel_backend="cuda",
            cache=cache
        ).eval()

    if unet_wrapper:
        pipe.unet = unet_wrapper(
            pipe.unet.state_dict(),
            quant_type=torch.float8_e4m3fn,
            dtype=args.dtype,
            kernel_backend="cuda"
        ).eval()

    if args.model_type == "flux":
        pipe.controlnet = FluxControlnetWrapper(
            pipe.controlnet.state_dict(),
            in_channels=64,
            quant_type=torch.float8_e4m3fn,
            kernel_backend="cuda"
        ).eval()
    else:  # sdxl
        pipe.controlnet = SDXLControlnetModelWrapper(
            pipe.controlnet.config,
            pipe.controlnet.state_dict(),
            dtype=args.dtype,
            quant_type=torch.float8_e4m3fn,
            kernel_backend="cuda"
        ).eval()

    # preprocess control image
    control_image = preprocess_fn(args.control_image)

    # common arguments for inference
    common_kwargs = dict(
        prompt=args.prompt,
        controlnet_conditioning_scale=args.controlnet_scale,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )

    # add image or control_image based on model type
    if args.model_type == "flux":
        common_kwargs["control_image"] = control_image
    else:  # sdxl
        common_kwargs["image"] = control_image
        common_kwargs["negative_prompt"] = args.negative_prompt

    # Warm-up
    pipe(**common_kwargs)
    torch.cuda.synchronize()

    # Timed inference
    start_time = time.time()
    images = pipe(**common_kwargs).images
    torch.cuda.synchronize()

    print(f"{args.model_type.upper()} FastDM inference time: {time.time() - start_time:.2f}s")
    images[0].save(args.output)

def str_to_dtype(dtype_str):
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise argparse.ArgumentTypeError(f"不支持的数据类型: {dtype_str}")
    return dtype_map[dtype_str]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["flux", "sdxl"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--controlnet_model", type=str, required=True)
    parser.add_argument("--vae_model", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--dtype", type=str_to_dtype, default=torch.bfloat16)
    parser.add_argument("--control_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="a beautiful landscape")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--controlnet_scale", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output", type=str, default="output.png")

    # caching args
    parser.add_argument('--cache-config', type=str, default=None, help="cache config json file path")

    args = parser.parse_args()

    if args.model_type == "flux":
        from diffusers import FluxControlNetModel, FluxControlNetPipeline
        run_pipeline(
            pipe_class=FluxControlNetPipeline,
            controlnet_class=FluxControlNetModel,
            transformer_wrapper=FluxTransformerWrapper,
            preprocess_fn=preprocess_flux,
            args=args
        )
    else:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
        run_pipeline(
            pipe_class=StableDiffusionXLControlNetPipeline,
            controlnet_class=ControlNetModel,
            unet_wrapper=SDXLUNetModelWrapper,
            preprocess_fn=preprocess_sdxl,
            args=args
        )
