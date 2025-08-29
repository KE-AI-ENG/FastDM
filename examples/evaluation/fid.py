import argparse
import torch

from PIL import Image
import os
import numpy as np
from torchvision.transforms import functional as F

# from fastdm.model.flux import FluxTransformerWrapper
from fastdm.model_entry import FluxTransformerWrapper, SD35TransformerWrapper, SDXLUNetModelWrapper, QwenTransformerWrapper
from diffusers import DiffusionPipeline

from torchmetrics.image.fid import FrechetInceptionDistance

import pathlib, glob, json
import pandas as pd

from multiprocessing import Process

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

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for compute fid", conflict_handler='resolve')
    parser.add_argument('--height', type=int, default=1024, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--width', type=int, default=2048, help="Height of image to generate (must be multiple of 8)")
    parser.add_argument('--steps', type=int, default=25, help="Inference steps")
    parser.add_argument('--max-seq-len', type=int, default=512, help="max sequence length")
    parser.add_argument('--guidance-scale', type=float, default=3.5, help="guidance scale")
    parser.add_argument('--seed', type=int, default=42, help="generation seed")
    
    parser.add_argument('--device', type=int, default=0, help="device number")
    parser.add_argument('--data-parallel', action='store_true', help="Enable multiGPU parallel")
    parser.add_argument('--devices', nargs='+', type=int, default=[0], help="Device numbers. Such as, 1 2 3")

    parser.add_argument('--prompts', nargs='+', type=str, default=["a photo of an astronaut riding a horse on mars"],
                        help='List for prompts, split by Space')
    parser.add_argument('--negative-prompts', type=str, default=None, help="negative text prompt")
    parser.add_argument("--enable-dataset", action="store_true", help="Enable dataset")
    parser.add_argument("--prompts-path", type=str, default="", help="Path for dataset, The file type is JSONL.")
    parser.add_argument('--real-images-path', type=str, default='', help="Directory for real images")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--class-id-path", type=str, default="", help="The json file path of class and id")

    parser.add_argument('--save-path', type=str, default=None, help="Path for images and csv")
    parser.add_argument('--use-diffusers', action='store_true', help="Use hf-diffusers, Enable pytorch scale-dot-product-attention")

    parser.add_argument('--enable-compile', action='store_true', help="Enable pytorch compile")

    parser.add_argument('--use-fp8', action='store_true', help="Enable fp8 model inference")
    parser.add_argument('--use-int8', action='store_true', help="Enable int8 model inference")
    parser.add_argument('--kernel-backend', default="cuda", help="kernel backend: torch/triton/cuda")

    parser.add_argument('--model-path', default='', help="Directory for SDXL model path")
    parser.add_argument('--architecture', default="sdxl", choices=['sdxl', 'sdxl-controlnet', 'flux', 'sd3'], help="model architecture: sdxl/sdxl-controlnet/flux/sd3")

    parser.add_argument("--enable-get-fid-error", action="store_true", help="Enable compute fid group error")
    parser.add_argument('--baseline-path', type=str, default='', help="Directory for baseline data")
    parser.add_argument('--quantization-path', type=str, default='', help="Directory for quantization data")
    parser.add_argument('--csv-path', type=str, default='', help="Directory for get-fid-error")
    
    return parser.parse_args()

def real_images_preprocess(dataset_path, height, width):
    image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (height, width))

    real_images = torch.cat([preprocess_image(image) for image in real_images])

    return real_images  # [10, 3, 256, 256]

def generate_images(args, prompt_text, quant_type, logical_id=0, rank_id=0, per_prompts_for_dev=0):
    # model config
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, use_safetensors=True)
    pipe.to("cuda")

    if not args.use_diffusers:
        if "sdxl" == args.architecture:
            pipe.unet = SDXLUNetModelWrapper(pipe.unet.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        elif "flux" == args.architecture:
            pipe.transformer = FluxTransformerWrapper(pipe.transformer.state_dict(), in_channels=64, quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        elif "sd3" == args.architecture:
            pipe.transformer = SD35TransformerWrapper(pipe.transformer.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        elif "qwen" == args.architecture:
            pipe.transformer = QwenTransformerWrapper(pipe.transformer.state_dict(), quant_type=quant_type, kernel_backend=args.kernel_backend).eval()
        else:
            raise ValueError(
                f"The {args.architecture} model is not supported!!!"
            )

    # generate images
    gen = torch.manual_seed(args.seed)
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
        if args.architecture == "xl":
            pipe_params["negative_prompt"] = args.negative_prompts

        images = pipe(**pipe_params).images
        images_list.append(images)
        
        # save
        if args.save_path:
            save_path = os.path.join(args.save_path, "images")
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True) 
            for j in range(images.shape[0]):
                image = Image.fromarray((images[j]*255).astype(np.uint8))
                image.save(f"{save_path}/rankid_{rank_id}_promptid_{pmt_idx(logical_id, i*args.batch_size+j, per_prompts_for_dev)}.png")
    
    if not args.data_parallel:
        output =  np.concatenate(images_list, axis=0)
        return torch.tensor(output).permute(0,3,1,2)    # [b, 3, 256, 256]


def get_fid(real_images, fake_images):
    fid = FrechetInceptionDistance(normalize=True)

    fid.update(real_images, real=True)  # tensor
    fid.update(fake_images, real=False)

    fid_ = float(fid.compute())

    return fid_


def worker(logical_id, rank_id, prompts, args, per_prompts_for_dev, quant_type):
    torch.cuda.set_device(rank_id)
    print(f"current_device:{torch.cuda.current_device()}, process id:{os.getpid()}")

    generate_images(args, prompts, quant_type, logical_id, rank_id, per_prompts_for_dev)


def load_prompts(prompts_path):
    prompts = []

    with open(prompts_path, 'r') as f:
        data = json.load(f)
        for i in range(len(data)):
            prompts.append(data[str(int(i)+1)])
    return prompts


def scatter_prompt(prompts, devices):
    devices.sort()
    prompts_list = []

    per_prompts_for_dev = (len(prompts) + len(devices) - 1) // len(devices)
    for i in range(0, len(prompts), per_prompts_for_dev):
        per_prompts = prompts[i:i+per_prompts_for_dev]
        prompts_list.append(per_prompts)

    return prompts_list, per_prompts_for_dev


def load_images(images_paths, height, width):
    """
    images_paths: list, eg: [path0, path1, ...]
    """
    images = []
    for img_path in images_paths:
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img)
        images.append(img)

    def preprocess_image(image):
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return F.center_crop(image, (height, width))

    images = torch.cat([preprocess_image(img) for img in images])

    return images   # [b, 3, height, width]


def pmt_idx(logical_id, id_in_device, per_prompts_for_dev):
    prompt_idx = logical_id * per_prompts_for_dev + id_in_device

    return f"{prompt_idx}"


def get_fid_group(real_images_path, gener_images_path, height, width, batch_fid=500):
    """
    Input the real image path and the generated image path, generate fid by group and generate csv files. 
    csv's path is the parent folder of gener_images_path, and the format is
    group_id    fid
    0           180.00
    """
    # grouping
    rimg_paths = sorted(
        os.listdir(real_images_path),
        key=lambda x: int(os.path.splitext(x.split("2012_val_")[-1])[0])
    )   # eg: ILSVRC2012_val_00000026.JPEG
    rimg_paths = [os.path.join(real_images_path, x) for x in rimg_paths]
    gener_images_path = os.path.join(gener_images_path, "images")
    gimg_paths = sorted(
        os.listdir(gener_images_path),
        key=lambda x: int(os.path.splitext(x.split("_promptid_")[-1])[0])
    )   # eg: rankid_1_promptid_0.png
    gimg_paths = [os.path.join(gener_images_path, x) for x in gimg_paths]

    rimg_group = []     # [list0, lsit2, ...]
    gimg_group = []
    for i in range(0, len(rimg_paths), batch_fid):
        if (i+batch_fid)>=len(rimg_paths) and len(rimg_paths[i:i+batch_fid])==1:    # the last group just have one element
            rimg = rimg_group[-1].pop()
            rlist = [rimg]
            rlist.extend(rimg_paths[i:i+batch_fid])
            rimg_group.append(rlist)
            gimg = gimg_group[-1].pop()
            glist = [gimg]
            glist.extend(gimg_paths[i:i+batch_fid])
            gimg_group.append(glist)
            break
        rimg_group.append(rimg_paths[i:i+batch_fid])
        gimg_group.append(gimg_paths[i:i+batch_fid])

    # load images and get fid
    fid_group = []

    for i in range(len(rimg_group)):
        real_images = load_images(rimg_group[i], height, width)    # tensor
        gener_images = load_images(gimg_group[i], height, width)
        fid_ = get_fid(real_images, gener_images)
        fid_group.append(fid_)

    # save fid
    group_id = [x for x in range(len(fid_group))]
    df = pd.DataFrame({"group_id":group_id, "fid": fid_group})
    gener_images_path = os.path.dirname(gener_images_path)
    df.to_csv(os.path.join(gener_images_path, "fid.csv"), index=False, sep=',')

   
def get_fid_group_error(baseline_path, quantization_path, csv_path):
    """
    load baseline_fid和quant_fid, compute MSE、MAE、Max Error、Min Error、Cumulative Error、
    Absolute Cumulative Error、mean_cumulative_error and mean_absolute_cumulative_error
    """
    # load fid
    baseline_fid_path = os.path.join(baseline_path, "fid.csv")
    quant_fid_path = os.path.join(quantization_path, "fid.csv")
    baseline = pd.read_csv(baseline_fid_path)["fid"]
    quantization = pd.read_csv(quant_fid_path)["fid"]

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
        "平均fid(baseline)": [mean_baseline],
        "平均fid(quantization)": [mean_quantization],
        "均方差损失MSE": [mse],
        "平均绝对误差MAE": [mae],
        "最大误差max_error": [max_error],
        "最小误差min_error": [min_error],
        "累计误差cumulative_error": [cumulative_error],
        "累计绝对误差absolute_cumulative_error": [absolute_cumulative_error],
        "平均误差mean_cumulative_error": [mean_cumulative_error],
        "平均累计误差mean_absolute_cumulative_error": [mean_absolute_cumulative_error],
    })
    df.to_csv(os.path.join(csv_path,"fid_error.csv"))

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

    if not args.enable_get_fid_error:
        # dataset
        if args.enable_dataset:
            prompts = load_prompts(args.prompts_path)    # [prompt0, prompt1, ...]
            if args.data_parallel:
                prompts, per_prompts_for_dev = scatter_prompt(prompts, args.devices) # [[prompt0,...], [], ...]
        else:
            prompts = args.prompts

        if args.data_parallel:
            torch.multiprocessing.set_start_method("spawn", force=True)
            print(f"current process id: {os.getpid()}")

            processes = []
            for i, rank_id in enumerate(args.devices):
                p = Process(target=worker, args=(i, rank_id, prompts[i], args, per_prompts_for_dev, quant_type))
                processes.append(p)
            
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            print("============================ All images are generated! ============================")

            get_fid_group(args.real_images_path, args.save_path, args.height, args.width)

        else:
            torch.cuda.set_device(args.device)
            
            # step1 import real images and pre-process
            real_images = real_images_preprocess(args.real_images_path, args.height, args.width)

            # step2 generate images
            prompts = [
                "cassette player",
                "chainsaw",
                "chainsaw",
                "church",
                "gas pump",
                "gas pump",
                "gas pump",
                "parachute",
                "parachute",
                "tench",
            ]
            gener_images = generate_images(args, prompts, quant_type)

            # step3 compute the fid
            fid_ = get_fid(real_images, gener_images)
            print(f"FID: {fid_}")
    
    else:
        get_fid_group_error(args.baseline_path, args.quantization_path, args.csv_path)