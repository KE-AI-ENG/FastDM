import argparse
import time
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers.utils import load_image
from diffusers import FluxControlNetModel, FluxControlNetPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from fastdm.model_entry import create_model
from fastdm.common_args import get_conrolnet_parser

from fastdm.caching.xcaching import AutoCache


def preprocess_sdxl(image_path):
    """SDXL: convert canny image"""
    image = np.array(load_image(image_path))
    image = cv2.Canny(image, 100, 200)
    image = np.repeat(image[:, :, None], 3, axis=2)
    return Image.fromarray(image)

def run_pipeline(args):
    # get quantization type
    if args.use_fp8:
        print("enable fp8 model inference!")
        quant_type = torch.float8_e4m3fn
    elif args.use_int8:
        print("enable int8 model inference!")
        quant_type = torch.int8
    else:
        quant_type = None

    # load controlnet pipeline
    if args.architecture == "flux-controlnet":
        controlnet = FluxControlNetModel.from_pretrained(
            args.controlnet_model,
            torch_dtype=args.data_type
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            args.model_path,
            controlnet=controlnet,
            torch_dtype=args.data_type
        ).to("cuda")
    elif args.architecture == "sdxl-controlnet":
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_model,
            torch_dtype=args.data_type
        )
        vae = AutoencoderKL.from_pretrained(
            args.vae_model, torch_dtype=args.data_type
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.model_path,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=args.data_type
        ).to("cuda")
    else:
        raise ValueError(f"Controlnet current only support flux and sdxl, but got {args.architecture}")

    # set cache
    if args.cache_config is not None:
        cache = AutoCache.from_json(args.cache_config)
        cache.config.current_steps_callback = lambda: pipe.scheduler.step_index
        cache.config.total_steps_callback = lambda: pipe.scheduler.timesteps.shape[0] # used by dicache
    else:
        cache = None

    if not args.use_diffusers:
        if args.architecture == "flux-controlnet":
            pipe.transformer = create_model(
                "flux",
                ckpt_path=pipe.transformer.state_dict(),
                in_channels=64,
                quant_type=quant_type,
                dtype=args.data_type,
                kernel_backend=args.kernel_backend,
                cache=cache
            ).eval()

            pipe.controlnet = create_model(
                "flux_controlnet",
                ckpt_path=pipe.controlnet.state_dict(),
                in_channels=64,
                quant_type=quant_type,
                kernel_backend=args.kernel_backend
            ).eval()
        elif args.architecture == "sdxl-controlnet":
            pipe.unet = create_model(
                "sdxl",
                ckpt_path=pipe.unet.state_dict(),
                quant_type=quant_type,
                dtype=args.data_type,
                kernel_backend=args.kernel_backend
            ).eval()

            pipe.controlnet = create_model(
                "sdxl_controlnet",
                ckpt_config=pipe.controlnet.config,
                ckpt_path=pipe.controlnet.state_dict(),
                dtype=args.data_type,
                quant_type=quant_type,
                kernel_backend=args.kernel_backend
            ).eval()

    # preprocess control image
    if args.architecture == "sdxl-controlnet":
        control_image = preprocess_sdxl(args.control_image_path)
    else:
        control_image = load_image(args.control_image_path)

    # common arguments for inference
    common_kwargs = dict(
        prompt=args.prompts,
        controlnet_conditioning_scale=args.controlnet_scale,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
    )

    # add image or control_image based on model type
    if args.architecture == "flux-controlnet":
        common_kwargs["control_image"] = control_image
    else:  # sdxl
        common_kwargs["image"] = control_image
        common_kwargs["negative_prompt"] = args.negative_prompts

    # Warm-up
    pipe(**common_kwargs)
    torch.cuda.synchronize()

    # Timed inference
    start_time = time.time()
    images = pipe(**common_kwargs).images
    torch.cuda.synchronize()

    print(f"{args.architecture.upper()} FastDM inference time: {time.time() - start_time:.2f}s")
    images[0].save(args.output_path)

def str_to_dtype(dtype_str):
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise argparse.ArgumentTypeError(f"不支持的数据类型: {dtype_str}")
    return dtype_map[dtype_str]

if __name__ == "__main__":
    args = get_conrolnet_parser().parse_args()
    args.data_type = str_to_dtype(args.data_type)
    run_pipeline(args)
