# usage:
# text2img
'''
python gen.py 
    --model-path /path/to/flux-model 
    --architecture flux 
    --height 1024 
    --width 2048 
    --steps 25 
    --use-fp8 
    --output-path ./flux-fp8.png
'''

# text2video
'''
python gen.py 
    --model-path /path/to/wan-model
    --prompts "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
    --architecture wan
    --height 704
    --width 1280
    --steps 50
    --num-frames 121
    --fps 24
    --guidance-scale 5.0
    --use-fp8
    --output-path wan.mp4
'''

import time
import gc
import argparse

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers import DiffusionPipeline, WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video

from fastdm.model_entry import FluxTransformerWrapper, SD35TransformerWrapper, SDXLUNetModelWrapper, QwenTransformerWrapper, WanTransformer3DWrapper
from fastdm.cache_config import CacheConfig
# import transformers
# transformers.utils.move_cache()

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for Diffusion model Txt2Img Demo", conflict_handler='resolve')
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=2048, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--num-frames', type=int, default=121, help="video frames to generate")
    parser.add_argument('--fps', type=int, default=24, help="output video fps")
    parser.add_argument('--steps', type=int, default=25, help="Inference steps")
    parser.add_argument('--max-seq-len', type=int, default=512, help="max sequence length")
    parser.add_argument('--guidance-scale', type=float, default=3.5, help="guidance scale")
    parser.add_argument('--seed', type=int, default=42, help="generation seed")
    parser.add_argument('--device', type=int, default=0, help="device number")
    parser.add_argument('--num-warmup-runs', type=int, default=1, help="Number of warmup runs before benchmarking performance")

    parser.add_argument('--prompts', type=str, default='An astronaut riding a horse', help="text prompt")
    parser.add_argument('--negative-prompts', type=str, default='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走', help="negative text prompt")

    parser.add_argument('--output-path', type=str, default=None, help="output png/mp4 path")

    parser.add_argument('--use-diffusers', action='store_true', help="Use hf-diffusers, Enable pytorch scale-dot-product-attention")

    parser.add_argument('--use-fp8', action='store_true', help="Enable fp8 model inference")
    parser.add_argument('--use-int8', action='store_true', help="Enable int8 model inference")
    parser.add_argument('--kernel-backend', default="cuda", help="kernel backend: torch/triton/cuda")

    parser.add_argument('--qwen-oom-resolve', action='store_true', help="It can resolve OOM error of qwen-image model if set to true")

    parser.add_argument('--model-path', default='', help="Directory for diffusion model path")

    parser.add_argument('--data-type', default="bfloat16", help="data type")
    parser.add_argument('--architecture', default="sdxl", help="model architecture: sdxl/flux/sd3/qwen/wan")

    # caching args
    parser.add_argument('--cache-config', type=str, default=None, help="cache config json file path")

    return parser.parse_args()

def pipe_running(pipe, prompt_text, negative_prompts, dump_img_path=None, gen_seed=42, inf_step=25, gen_width=2048, gen_height=1024, warmup=1, max_len=512, guidance=3.5, num_frames=121, fps=24, video_gen=False):

    gen = torch.Generator().manual_seed(gen_seed)

    #warm up
    for i in range(warmup):
        if video_gen:
            print(f"warmup {i+1}/{warmup} for video generation")
            frame = pipe(prompt=prompt_text, negative_prompt=negative_prompts, num_inference_steps=inf_step, generator=gen, width=gen_width, height=gen_height, guidance_scale=guidance, max_sequence_length=max_len, num_frames=num_frames).frames[0]
        else:
            print(f"warmup {i+1}/{warmup} for image generation")
            images = pipe(prompt=prompt_text, num_inference_steps=inf_step, generator=gen, width=gen_width, height=gen_height, guidance_scale=guidance, max_sequence_length=max_len).images[0]
        torch.cuda.synchronize()

    start_ = time.time()
    if video_gen:
        print("start video generation...")
        frame = pipe(prompt=prompt_text, negative_prompt=negative_prompts, num_inference_steps=inf_step, generator=gen, width=gen_width, height=gen_height, guidance_scale=guidance, max_sequence_length=max_len, num_frames=num_frames).frames[0]
        torch.cuda.synchronize()
        print(f"inference time: {time.time() - start_}s")
        if dump_img_path is not None:
            export_to_video(frame, dump_img_path, fps=fps)
    else:
        print("start image generation...")
        images = pipe(prompt=prompt_text, num_inference_steps=inf_step, generator=gen, width=gen_width, height=gen_height, guidance_scale=guidance, max_sequence_length=max_len).images[0]
        torch.cuda.synchronize()
        print(f"inference time: {time.time() - start_}s")
        if dump_img_path is not None:
            images.save(dump_img_path)

    return

if __name__ == "__main__":

    args = parseArgs()

    running_data_type = torch.float16 if "float16"==args.data_type else torch.bfloat16

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

    if args.architecture == "wan":
        vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16)
    else:
        pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype = running_data_type, use_safetensors=True)

    video_gen_ = args.architecture == "wan"
    
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    if args.cache_config is not None:
        cache_config = CacheConfig.from_json(args.cache_config)
        cache_config.current_steps_callback = lambda: pipe.scheduler.step_index
    else:
        cache_config = None

    if args.use_diffusers:
        pass
    else:
        if "sdxl" == args.architecture:
            pipe.unet = SDXLUNetModelWrapper(pipe.unet.state_dict(), dtype=running_data_type, quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        elif "flux" == args.architecture:
            pipe.transformer = FluxTransformerWrapper(pipe.transformer.state_dict(), dtype=running_data_type, in_channels=64, quant_type=quant_type, kernel_backend=args.kernel_backend,
                                                    cache_config=cache_config).eval()
        elif "sd3" == args.architecture:
            pipe.transformer = SD35TransformerWrapper(pipe.transformer.state_dict(), dtype=running_data_type, quant_type=quant_type, kernel_backend=args.kernel_backend,
                                                      cache_config=cache_config).eval()
        elif "qwen" == args.architecture:
            pipe.transformer = QwenTransformerWrapper(pipe.transformer.state_dict(), dtype=running_data_type, quant_type=quant_type, kernel_backend=args.kernel_backend,
                                                      cache_config=cache_config, need_resolve_oom=args.qwen_oom_resolve).eval()
            if args.qwen_oom_resolve:
                import os
                import sys
                from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
                sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
                from utils.qwen_vae import qwen_vae_new_decode, qwen_vae_new_encode
                AutoencoderKLQwenImage._decode = qwen_vae_new_decode
                AutoencoderKLQwenImage._encode = qwen_vae_new_encode
        elif "wan" == args.architecture:
            pipe.transformer = WanTransformer3DWrapper(pipe.transformer.state_dict(), dtype=running_data_type, quant_type=quant_type, kernel_backend=args.kernel_backend, config_json=f"{args.model_path}/transformer/config.json",
                                                    cache_config=cache_config).eval()
            if hasattr(pipe, 'transformer_2') and pipe.transformer_2 is not None:
                pipe.transformer_2 = WanTransformer3DWrapper(pipe.transformer_2.state_dict(), dtype=running_data_type, quant_type=quant_type, kernel_backend=args.kernel_backend, config_json=f"{args.model_path}/transformer_2/config.json",
                                            cache_config=cache_config).eval()
        else:
            raise ValueError(
                f"The {args.architecture} model is not supported!!!"
            )

    if "qwen" == args.architecture and args.qwen_oom_resolve:
        pipe.vae.to("cuda")
    else:
        pipe.to("cuda")

    torch.cuda.empty_cache()
    gc.collect()

    if video_gen_:
        assert args.output_path.endswith('.mp4'), "output path must end with .mp4 for video generation"

    pipe_running(pipe, args.prompts, args.negative_prompts, args.output_path, args.seed, args.steps, args.width, args.height, args.num_warmup_runs, args.max_seq_len, args.guidance_scale, args.num_frames, args.fps, video_gen=video_gen_)

    print(f'mem-usage: {(torch.cuda.memory_allocated(torch.cuda.current_device()))/1024/1024/1024}GB')
    #print(f'max mem-usage: {(torch.cuda.max_memory_allocated(torch.cuda.current_device()))/1024/1024/1024}GB')

    #prof.export_chrome_trace("trace.json")