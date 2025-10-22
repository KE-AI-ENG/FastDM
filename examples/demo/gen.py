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
import json
import os

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers.utils import export_to_video

from fastdm.common_args import get_text_gen_parser
from fastdm.model_entry import FastDMEngine

if __name__ == "__main__":
    args = get_text_gen_parser().parse_args()

    if args.use_diffusers:
        pass
    else:
        print(f"*************use kernel backend: {args.kernel_backend}**************")

        if args.use_fp8:
            print("enable fp8 model inference!")
        elif args.use_int8:
            print("enable int8 model inference!")
    
    if args.image_path is not None:
        assert args.task in ["i2i", "i2v"], "Image path is only valid for i2i or i2v tasks"
    
    if args.lora_config is not None:
        assert args.architecture in ["qwen"], "LoRA is only supported for Qwen architecture currently"
        try:
            if os.path.isfile(args.lora_config):
                with open(args.lora_config, 'r') as f:
                    lora_config = json.load(f)
            else:
                lora_config = json.loads(args.lora_config)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in Lora config: {args.lora_config}")
        lora_name = next(iter(lora_config), None) # only support one lora model in gen.py

    model_load_start = time.time()
    engine = FastDMEngine(
        model_path=args.model_path,
        architecture=args.architecture,
        device=args.device,
        data_type=args.data_type,
        use_fp8=args.use_fp8,
        use_int8=args.use_int8,
        kernel_backend=args.kernel_backend,
        cache_config=args.cache_config,
        oom_resolve=args.oom_resolve,
        use_diffusers=args.use_diffusers,
        task=args.task,
        sparse_attn_config=args.sparse_attn_config,
        lora_config=lora_config if args.lora_config is not None else None,
    )
    model_load_time = time.time() - model_load_start
    print(f"Model loading latency: {model_load_time:.4f} seconds")
    
    # 预热
    for i in range(args.num_warmup_runs):
        print(f"warmup {i+1}/{args.num_warmup_runs}")
        engine.generate(
            prompt=args.prompts,
            negative_prompt=args.negative_prompts,
            num_frames=args.num_frames if args.task == "t2v" or args.task == "i2v" else None,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            gen_seed=args.seed,
            gen_width=args.width,
            gen_height=args.height,
            max_seq_len=args.max_seq_len,
            true_cfg_scale= args.true_cfg_scale if "qwen" == args.architecture else None,
            src_image=args.image_path if args.task == "i2v" else None,
            lora_name=lora_name if args.lora_config is not None else None
        )
    
    # 生成
    gen_start_time = time.time()
    output = engine.generate(
                prompt=args.prompts,
                negative_prompt=args.negative_prompts,
                num_frames=args.num_frames if args.task == "t2v" or args.task == "i2v" else None,
                steps=args.steps,
                guidance_scale=args.guidance_scale,
                gen_seed=args.seed,
                gen_width=args.width,
                gen_height=args.height,
                max_seq_len=args.max_seq_len,
                true_cfg_scale= args.true_cfg_scale if "qwen" == args.architecture else None,
                src_image=args.image_path if args.task == "i2v" else None,
                lora_name=lora_name if args.lora_config is not None else None)
    torch.cuda.synchronize()
    generation_time = time.time() - gen_start_time
    print(f"Generation latency: {generation_time:.4f} seconds")

    # 保存结果
    if args.task in ["t2v", "i2v"]:
        export_to_video(output, args.output_path, fps=args.fps)
    else:
        output.save(args.output_path)

    print(f'mem-usage: {(torch.cuda.memory_allocated(torch.cuda.current_device()))/1024/1024/1024}GB')
    print(f'max mem-usage: {(torch.cuda.max_memory_allocated(torch.cuda.current_device()))/1024/1024/1024}GB')

    #prof.export_chrome_trace("trace.json")