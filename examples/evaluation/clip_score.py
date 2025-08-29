from fastdm.model_entry import FluxTransformerWrapper, SD35TransformerWrapper, SDXLUNetModelWrapper, QwenTransformerWrapper
from diffusers import DiffusionPipeline

import torch
import argparse
import os
import json
import pandas as pd
import numpy as np
from PIL import Image
import pathlib
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from multiprocessing import Process

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for compute clip_score", conflict_handler='resolve')
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=2048, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--steps', type=int, default=25, help="Inference steps")
    parser.add_argument('--seed', type=int, default=42, help="generation seed")

    parser.add_argument('--max-seq-len', type=int, default=512, help="max sequence length")
    parser.add_argument('--guidance-scale', type=float, default=3.5, help="guidance scale")

    parser.add_argument('--device', type=int, default=0, help="device number")
    parser.add_argument('--data-parallel', action='store_true', help="Enable multiGPU parallel")
    parser.add_argument('--devices', nargs='+', type=int, default=[0], help="Device numbers. Such as, 1 2 3")

    parser.add_argument('--prompts', type=str, default='An astronaut riding a horse', help="text prompt")
    parser.add_argument('--negative-prompts', type=str, default=None, help="negative text prompt")
    parser.add_argument("--enable-dataset", action="store_true", help="Enable dataset")
    parser.add_argument("--dataset-path", type=str, default="", help="Path for dataset, The file type is JSONL.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    parser.add_argument('--save-path', type=str, default=None, help="Path for images and csv. Such as, ./images")
    parser.add_argument('--use-diffusers', action='store_true', help="Use hf-diffusers, Enable pytorch scale-dot-product-attention")

    parser.add_argument('--enable-compile', action='store_true', help="Enable pytorch compile")

    parser.add_argument('--use-fp8', action='store_true', help="Enable fp8 model inference")
    parser.add_argument('--use-int8', action='store_true', help="Enable int8 model inference")
    parser.add_argument('--kernel-backend', default="cuda", help="kernel backend: torch/triton/cuda")

    parser.add_argument('--model-path', default='', help="Directory for SDXL model path")
    parser.add_argument('--architecture', default="sdxl", choices=['sdxl', 'sdxl-controlnet', 'flux', 'sd3'], help="model architecture: sdxl/sdxl-controlnet/flux/sd3")

    parser.add_argument('--validate-model-path', default='', help="Directory for validation model path")

    parser.add_argument("--compute-clip-score-error", action="store_true")
    parser.add_argument("--baseline-clip-score-path", type=str, default="", help="Path for baseline's clip_score. Such as, ./")
    parser.add_argument("--quant-clip-score-path", type=str, default="", help="Path for quantization's clip_score. Such as, ./")
    parser.add_argument("--clip-score-error-path", type=str, default="", help="Path for clip_score_error's clip_score. Such as, ./")

    return parser.parse_args()

def mean_squared_error(y_true, y_pred):
    """
    计算均方误差 (MSE)
    参数:
    y_true -- 真实值列表/数组
    y_pred -- 预测值列表/数组
    
    返回:
    mse -- 均方误差值
    """
    # 检查输入长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError("输入数组长度不一致")
    
    # 计算平方误差和
    squared_errors = [(actual - predicted) ** 2 for actual, predicted in zip(y_true, y_pred)]
    
    # 计算平均值
    mse = sum(squared_errors) / len(y_true)
    return mse

def mean_absolute_error(y_true, y_pred):
    """
    计算平均绝对误差 (MAE)
    参数:
    y_true -- 真实值列表/数组
    y_pred -- 预测值列表/数组
    
    返回:
    mae -- 平均绝对误差值
    """
    # 检查输入长度是否一致
    if len(y_true) != len(y_pred):
        raise ValueError("输入数组长度不一致")
    
    # 计算绝对误差和
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(y_true, y_pred)]
    
    # 计算平均值
    mae = sum(absolute_errors) / len(y_true)
    return mae

def scatter_dataset(prompts, devices):
    """
    Slice the dataset for each device.
    Return: List, the element is prompt for each device.
    """
    devices.sort()
    prompts_list = []

    remainder = len(prompts) % len(devices)
    per_prompts_for_device = len(prompts) // len(devices) + (0 if remainder==0 else 1)
    start = 0
    for i in range(len(devices)):
        end = min(start + per_prompts_for_device, len(prompts))
        prompts_list.append(prompts[start:end])
        start = end

    return prompts_list, per_prompts_for_device, remainder

def model_config(args, quant_type):
    if args.architecture == "flux":
        sd_pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, use_safetensors=True)
    else:
        sd_pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    sd_pipe.to("cuda")

    if not args.use_diffusers:        
        if "sdxl" == args.architecture:
            sd_pipe.unet = SDXLUNetModelWrapper(sd_pipe.unet.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        elif "flux" == args.architecture:
            sd_pipe.transformer = FluxTransformerWrapper(sd_pipe.transformer.state_dict(), in_channels=64, quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        elif "sd3" == args.architecture:
            sd_pipe.transformer = SD35TransformerWrapper(sd_pipe.transformer.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        elif "qwen" == args.architecture:
            sd_pipe.transformer = QwenTransformerWrapper(sd_pipe.transformer.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        else:
            raise ValueError(
                f"The {args.architecture} model is not supported!!!"
            )

    return sd_pipe

def model_running(sd_pipe, prompt_text, args, rank_id, logical_id, per_pmt_for_device):
    # logical_id is the sorted index of the gpu
    gen = torch.Generator().manual_seed(args.seed)
    
    images_list = []
    for i in range(0, len(prompt_text), args.batch_size):
        len_per_prompt = len(prompt_text[i:i + args.batch_size])
        pipe_params = {
            "prompt": [""] * len_per_prompt,
            "prompt_2": prompt_text[i:i+args.batch_size],
            "num_inference_steps": args.steps,
            "generator": gen,
            "width": args.width,
            "height": args.height,
            "output_type": "np",
            "guidance_scale": args.guidance_scale, 
            "max_sequence_length": args.max_seq_len
        }

        images = sd_pipe(**pipe_params).images  # float
        images_list.append(images)
        
        if args.save_path:
            save_path = os.path.join(args.save_path, "images")
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
            for j in range(images.shape[0]):
                image = Image.fromarray((images[j]*255).astype(np.uint8))   # int
                image.save(f"{save_path}/rankid_{rank_id}_promptid_{pmt_idx(logical_id, i*args.batch_size+j, per_pmt_for_device)}.png")
    
    return np.concatenate(images_list, axis=0)  # [b,h,w,3]


def load_images(images_paths):
    """
    images_paths: List, [path0, path1, ...]
    """
    images = []
    for img_path in images_paths:
        img = Image.open(img_path)  # [h,w,3]
        img = np.asarray(img)
        images.append(img)  # [[h,w,3], [h,w,3], ...]

    return images   # [[h,w,3], [h,w,3], ...]

def compute_clip_score_mean(images, prompt_text, args, is_load = False):
    clip_score_fn = partial(clip_score, model_name_or_path=args.validate_model_path)

    def calculate_clip_score(images, prompts, is_load = False):
        """
        is_load = False:    images: type=array, shape=[b,h,w,3], range=[0,1]
        is_load = True:     images: type=list, len = b, images[0].shape=[h,w,3], range=[0,255]
        """
        if not is_load:
            images_int = (images * 255).astype("uint8") # int
            clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
        else:
            images_int = images
            batch_img = 50
            clip_score_sum = 0

            for i in range(0, len(images), batch_img):
                per_images = np.stack(images_int[i:i+batch_img])
                per_prompts = prompts[i:i+batch_img]
                per_len = len(per_images)
                clip_score_sum += clip_score_fn(torch.from_numpy(per_images).permute(0, 3, 1, 2), per_prompts).detach() * per_len
            clip_score = clip_score_sum / len(images)

        return round(float(clip_score), 4)

    clip_score_ = calculate_clip_score(images, prompt_text, is_load=is_load)

    return clip_score_

def compute_clip_score_group(prompts, images):
    clip_scores = []
    clip_score_fn = partial(clip_score, model_name_or_path=args.validate_model_path)
    def calculate_clip_score(image, prompt):
        """
        images: type=array, shape=[h,w,3], range=[0,1]
        """
        clip_score = clip_score_fn(torch.tensor(image).permute(2, 0, 1), prompt).detach()
        return round(float(clip_score), 4)

    for i in range(len(prompts)):
        clip_score_ = calculate_clip_score(images[i], prompts[i])
        clip_scores.append(clip_score_)

    return clip_scores


def worker(logical_id, rank_id, prompts, args, per_pmt_for_device, quant_type):
    # logical_id is the sorted index of the gpu
    torch.cuda.set_device(rank_id)
    print(f"current_device:{torch.cuda.current_device()}, process id:{os.getpid()}")

    sd_pipe = model_config(args, quant_type)
    images = model_running(sd_pipe, prompts, args, rank_id, logical_id, per_pmt_for_device)
    
def pmt_idx(rank_id, id_in_batch, per_pmt_for_device):
    """
    Calculates the global id of prompt.
    """
    prompt_idx = rank_id * per_pmt_for_device + id_in_batch
    return f"{prompt_idx}"

def compute_clip_score(prompts, save_path, batch_clip_score = 50):
    """
    Input the prompts path and the generated image path, generate fid by group and generate csv files. 
    csv's path is the parent folder of gener_images_path, and the format is
    prompt_id   clip_score
    0           30.00
    """
    clip_score = []
    # grouping
    images_path = os.path.join(save_path, "images") # Path for images
    image_paths = sorted(   # PathS for all images
        os.listdir(images_path),
        key=lambda x: int(os.path.splitext(x.split("promptid_")[-1])[0])
    )
    for i in range(0, len(image_paths), batch_clip_score):
        per_image_paths = [ os.path.join(images_path, fn) for fn in image_paths[i:i+batch_clip_score]]
        per_images = load_images(per_image_paths)
        per_prompts = prompts[i:i+batch_clip_score]
        per_clip_score = compute_clip_score_group(per_prompts, per_images)
        clip_score.extend(per_clip_score)

    # save clip_score
    prompt_id = [x for x in range(len(image_paths))]
    df = pd.DataFrame({'prompt_id':prompt_id, 'clip_score':clip_score})
    df.to_csv(os.path.join(save_path, "clip_score.csv"), index=False, sep=',')


def compute_clip_score_error(baseline_clip_score_path, quant_clip_score_path, clip_score_error_path):
    """
    load baseline_clip_score and quant_clip_score, compute MSE、MAE、Max Error、Min Error、
    Cumulative Error、Absolute Cumulative Error、mean_cumulative_error and mean_absolute_cumulative_error
    """
    # load clip_score
    baseline_clip_score_path = os.path.join(baseline_clip_score_path, "clip_score.csv")
    quant_clip_score_path = os.path.join(quant_clip_score_path, "clip_score.csv")
    baseline = pd.read_csv(baseline_clip_score_path)["clip_score"]
    quantization = pd.read_csv(quant_clip_score_path)["clip_score"]

    # compute
    errors = baseline - quantization    # y_true - y_pred

    mean_baseline = baseline.mean()
    mean_quantization = quantization.mean()
    mse = mean_squared_error(baseline, quantization)
    mae = mean_absolute_error(baseline, quantization)
    max_error = np.max(errors)
    min_error = np.min(errors)
    cumulative_error = errors.sum()
    absolute_cumulative_error = np.abs(errors).sum()
    mean_cumulative_error = errors.mean()
    mean_absolute_cumulative_error = np.abs(errors).mean()

    # save
    df = pd.DataFrame({
        "平均clip_score(baseline)": [mean_baseline],
        "平均clip_score(quantization)": [mean_quantization],
        "均方差损失MSE": [mse],
        "平均绝对误差MAE": [mae],
        "最大误差max_error": [max_error],
        "最小误差min_error": [min_error],
        "累计误差cumulative_error": [cumulative_error],
        "累计绝对误差absolute_cumulative_error": [absolute_cumulative_error],
        "平均误差mean_cumulative_error": [mean_cumulative_error],
        "平均累计误差mean_absolute_cumulative_error": [mean_absolute_cumulative_error],
    })
    df.to_csv(os.path.join(clip_score_error_path,"clip_score_error.csv"))

    return mean_baseline, mean_quantization, mse, mae, max_error, min_error, cumulative_error, absolute_cumulative_error, mean_cumulative_error, mean_absolute_cumulative_error


if __name__ == "__main__":
    
    args = parseArgs()

    print(f"*************use kernel backend: {args.kernel_backend}**************")
    print(f"*************use gpu: {torch.cuda.get_device_name(0)}**************")
    print(f"*************use model: {args.architecture}**************")

    if args.use_fp8:
        print("enable fp8 model inference!")
        quant_type = torch.float8_e4m3fn
    elif args.use_int8:
        print("enable int8 model inference!")
        quant_type = torch.int8
    else:
        quant_type = None

    if not args.compute_clip_score_error:
        # dataset
        if args.enable_dataset:
            prompts = []
            with open(args.dataset_path, 'r') as f:     
                if "imagenet-1k" in args.dataset_path: # imagenet-1k
                    data = json.load(f)
                    for i in range(len(data)):
                        prompts.append(data[str(i+1)])
                else:   # k-mktr/improved-flux-prompts-photoreal-portrait
                    for line in f:
                        json_obj = json.loads(line)
                        prompts.append(json_obj['prompt'])

            if args.data_parallel:
                prompts_list, per_pmt_for_device, remainder = scatter_dataset(prompts, args.devices)

        else:
            prompts = [args.prompts]

        if args.data_parallel:
            torch.multiprocessing.set_start_method("spawn", force=True)
            print(f"perant process id:{os.getpid()}")

            processes = []  # Create process
            for i,rank_id in enumerate(args.devices):
                p = Process(target=worker, args=(i, rank_id, prompts_list[i], args, per_pmt_for_device, quant_type))  
                processes.append(p)
            
            for p in processes: # Start process 
                p.start()
            for p in processes: # Wait process
                p.join()
            print("============================ All images are generated! ============================")

            compute_clip_score(prompts, args.save_path)
            
        else:
            torch.cuda.set_device(args.device)
            sd_pipe = model_config(args, quant_type)
            images = model_running(sd_pipe, prompts, args, args.device, 0, len(prompts))
            clip_score_mean = compute_clip_score_mean(images, prompts, args)
            print(f"clip_score_mean: {clip_score_mean}")

    else:
        mean_baseline, mean_quantization, mse, mae, max_error, min_error, cumulative_error, absolute_cumulative_error, mean_cumulative_error, mean_absolute_cumulative_error = compute_clip_score_error(args.baseline_clip_score_path, args.quant_clip_score_path, args.clip_score_error_path)
        print(f"""
mean_baseline: {mean_baseline}
mean_quantization: {mean_quantization}
mse: {mse}
mae: {mae}
max_error: {max_error}
min_error: {min_error}
cumulative_error: {cumulative_error}
absolute_cumulative_error: {absolute_cumulative_error}
mean_cumulative_error: {mean_cumulative_error}
mean_absolute_cumulative_error: {mean_absolute_cumulative_error}""")