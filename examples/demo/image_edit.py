import os
import time
import argparse

import gc

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

from fastdm.model_entry import QwenTransformerWrapper, FluxTransformerWrapper
from fastdm.cache_config import CacheConfig

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Diffusion model Image Edit Demo", conflict_handler='resolve')
    parser.add_argument('--steps', type=int, default=25, help="Inference steps")
    parser.add_argument('--true-cfg-scale', type=float, default=4.0, help="true cfg scale")
    parser.add_argument('--guidance-scale', type=float, default=2.5, help="guidance scale")
    parser.add_argument('--seed', type=int, default=0, help="generation seed")
    parser.add_argument('--device', type=int, default=0, help="device number")
    parser.add_argument('--num-warmup-runs', type=int, default=0, help="Number of warmup runs before benchmarking performance")

    parser.add_argument('--prompts', type=str, default="Change the horse's color to purple, with a flash light background.", help="text prompt")
    parser.add_argument('--negative-prompts', type=str, default=None, help="negative text prompt")
    parser.add_argument('--image-path', type=str, default="./src-img.png", help="input image path")

    parser.add_argument('--output-path', type=str, default='./output_image_edit.png', help="output image path")

    parser.add_argument('--use-diffusers', action='store_true', help="Use hf-diffusers, Enable pytorch scale-dot-product-attention")

    parser.add_argument('--use-fp8', action='store_true', help="Enable fp8 model inference")
    parser.add_argument('--use-int8', action='store_true', help="Enable int8 model inference")
    parser.add_argument('--kernel-backend', default="cuda", help="kernel backend: torch/triton/cuda")

    parser.add_argument('--qwen-oom-resolve', action='store_true', help="It can resolve OOM error of qwen-image model if set to true")

    parser.add_argument('--model-path', default='', help="Directory for diffusion model path")

    # caching args
    parser.add_argument('--cache-config', type=str, default=None, help="cache config json file path")

    return parser.parse_args()

if __name__ == "__main__":

    args = parseArgs()

    print(f"*************use kernel backend: {args.kernel_backend}**************")

    if args.use_fp8:
        print("enable fp8 model inference!")
        quant_type = torch.float8_e4m3fn
    elif args.use_int8:
        print("enable int8 model inference!")
        quant_type = torch.int8
    else:
        quant_type = None

    torch.cuda.set_device(args.device)

    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    #pipe.enable_model_cpu_offload()

    is_qwen_img = False
    if "QwenImage" in pipe.transformer.config._class_name:
        is_qwen_img = True

    src_image = load_image(args.image_path)
    
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    if args.cache_config is not None:
        cache_config = CacheConfig.from_json(args.cache_config)
        cache_config.current_steps_callback = lambda: pipe.scheduler.step_index
    else:
        cache_config = None

    if args.use_diffusers:
        pass
    else:
        if is_qwen_img: #qwen image edit
            pipe.transformer = QwenTransformerWrapper(pipe.transformer.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend,
                                                      cache_config=cache_config, need_resolve_oom=args.qwen_oom_resolve).eval()
            if args.qwen_oom_resolve:
                import os
                import sys
                from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
                sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
                from utils.qwen_vae import qwen_vae_new_decode, qwen_vae_new_encode
                AutoencoderKLQwenImage._decode = qwen_vae_new_decode
                AutoencoderKLQwenImage._encode = qwen_vae_new_encode
        else: #/FLUX.1-Kontext-dev
            pipe.transformer = FluxTransformerWrapper(pipe.transformer.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend,
                                                      cache_config=cache_config).eval()

    if is_qwen_img and args.qwen_oom_resolve:
        pipe.vae.to("cuda")
    else:
        pipe.to(f"cuda")

    torch.cuda.empty_cache()
    gc.collect()

    gen = torch.manual_seed(args.seed)

    #warm up
    for i in range(args.num_warmup_runs):
        print(f"warmup {i+1}/{args.num_warmup_runs} for image editing")
        with torch.inference_mode():
            if is_qwen_img:#qwen image edit
                output_image = pipe(prompt=args.prompts, negative_prompt=args.negative_prompts, image=src_image, num_inference_steps=args.steps, generator=gen, true_cfg_scale=args.true_cfg_scale).images[0]
            else:#/FLUX.1-Kontext-dev
                output_image = pipe(prompt=args.prompts, negative_prompt=args.negative_prompts, image=src_image, num_inference_steps=args.steps, generator=gen, guidance_scale=args.guidance_scale).images[0]
        torch.cuda.synchronize()

    print("start image editing...")
    start_time = time.time()
    with torch.inference_mode():
        if is_qwen_img:#qwen image edit
            output_image = pipe(prompt=args.prompts, negative_prompt=args.negative_prompts, image=src_image, num_inference_steps=args.steps, generator=gen, true_cfg_scale=args.true_cfg_scale).images[0]
        else:#/FLUX.1-Kontext-dev
            output_image = pipe(prompt=args.prompts, negative_prompt=args.negative_prompts, image=src_image, num_inference_steps=args.steps, generator=gen, guidance_scale=args.guidance_scale).images[0]
    torch.cuda.synchronize()
    print(f"inference time: {time.time() - start_time}s")

    output_image.save(args.output_path)
    print("image saved at", os.path.abspath(args.output_path))

    #prof.export_chrome_trace("trace.json")