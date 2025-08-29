'''
This script is used to merge the lora weights into the flux transformer model weights. 
It will save a new transformer model with the lora weights merged into it.

# usage: python flux_lora_merge.py --model-path <flux_model_path> --lora-path <lora_model_path> --output-path <output_model_path>
'''

import os
import shutil
import torch
import argparse

from safetensors import safe_open
from safetensors.torch import save_file
from diffusers import DiffusionPipeline

def parseArgs():
    parser = argparse.ArgumentParser(description="Merge lora to flux transformer model", conflict_handler='resolve')
    parser.add_argument('--model-path', default='', required=True, help="Directory for flux model path")
    parser.add_argument('--lora-path', default='', required=True, help="Directory for lora model path")
    parser.add_argument('--merged-model-path', help="Directory for output model path")
    parser.add_argument('--combined-shard', action='store_true', help="Save the model as a single shard")
    parser.add_argument('--convert-lora', action='store_true', help="Convert lora weights to the new format")

    parser.add_argument('--data-type', default="bfloat16", help="data type")

    return parser.parse_args()

def lora_ckpt_conversion(lora_path):
    ori_tensors = {}
    dst_tensors = {}
    safetensors_path = None

    for root, dirs, files in os.walk(lora_path):
        for file in files:
            if file.endswith(".safetensors"):
                safetensors_path = os.path.join(root, file)
                print(f"Found lora safetensors file: {safetensors_path}")
                break

    with safe_open(safetensors_path, framework="pt") as f:
        for key in f.keys():
            ori_tensors[key] = f.get_tensor(key)

    for key,value in ori_tensors.items():
        dst_key = f"transformer.{key}"
        if "lora_A" in dst_key:
            dst_key = dst_key.replace("lora_A.default.weight","lora.down.weight")
        elif "lora_B" in dst_key:
            dst_key = dst_key.replace("lora_B.default.weight","lora.up.weight")
        else:
            print(f"Warning: {key} is not a lora weight, skip it!")

        dst_tensors[dst_key] = value
    
    new_safetensors_path = os.path.join(lora_path, "converted.safetensors")
    save_file(dst_tensors, new_safetensors_path)
    return new_safetensors_path

def megrge_and_save(args):
    if args.merged_model_path is None:
        args.merged_model_path = f"{args.model_path}-lora-merged"

    if os.path.exists(args.merged_model_path):
        print("Merged model path already exists, will delete it and create a new one")
        shutil.rmtree(args.merged_model_path)
        os.makedirs(args.merged_model_path)
    else:
        os.makedirs(args.merged_model_path)

    #formatting
    if args.convert_lora:
        safetensor_path = lora_ckpt_conversion(args.lora_path)
    else:
        safetensor_path = args.lora_path

    # Load the original model
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=args.data_type).to("cuda")
    pipe.load_lora_weights(safetensor_path, adapter_name="lora")
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    # Save the merged model
    pipe.save_pretrained(args.merged_model_path)
    print("Merged model saved to {}".format(args.merged_model_path))

def combined_transformer(new_model_path):
    '''
    For comfyui, merge transformer model weights to one file
    '''
    state_dict = {}
    transformer_path = os.path.join(new_model_path, "transformer")
    one_shard_transformer_path = os.path.join(transformer_path, "diffusion_pytorch_model.safetensors")
    print("Saving transformer model to one file {}".format(one_shard_transformer_path))
    for filename in sorted(os.listdir(transformer_path)):
        if filename.endswith(".safetensors"):
            with safe_open(os.path.join(transformer_path, filename), framework="pt") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    save_file(state_dict, one_shard_transformer_path, metadata={"format": "pt"})
    
if __name__ == "__main__":

    args = parseArgs()
    if args.data_type == "bfloat16":
        args.data_type = torch.bfloat16
    else:
        args.data_type = torch.float16
    print("Args: ", args)

    megrge_and_save(args)

    if args.combined_shard:
        combined_transformer(args.merged_model_path)