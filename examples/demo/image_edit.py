import os
import time

import torch
from diffusers.utils import load_image

from fastdm.common_args import get_image_edit_parser
from fastdm.model_entry import FastDMEngine

if __name__ == "__main__":

    args = get_image_edit_parser().parseArgs()

    if args.use_diffusers:
        pass
    else:
        print(f"*************use kernel backend: {args.kernel_backend}**************")

        if args.use_fp8:
            print("enable fp8 model inference!")
            quant_type = torch.float8_e4m3fn
        elif args.use_int8:
            print("enable int8 model inference!")
            quant_type = torch.int8
        else:
            quant_type = None

    # 加载模型
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
        use_diffusers=args.use_diffusers
    )
    model_load_time = time.time() - model_load_start
    print(f"Model loading time: {model_load_time:.4f} seconds")

    # 加载源图片
    src_image = load_image(args.image_path)

    # 预热
    for i in range(args.num_warmup_runs):
        print(f"warmup {i+1}/{args.num_warmup_runs}")
        engine.generate(
            prompt=args.prompts,
            negative_prompt=args.negative_prompts,
            image=src_image,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            gen_seed=args.seed
        )

    # 生成
    gen_start_time = time.time()
    output = engine.generate(
        prompt=args.prompts,
        negative_prompt=args.negative_prompts,
        image=src_image,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        gen_seed=args.seed
    )
    torch.cuda.synchronize()
    print(f"inference time: {time.time() - gen_start_time}s")

    output.save(args.output_path)
    print("image saved at", os.path.abspath(args.output_path))

    #prof.export_chrome_trace("trace.json")