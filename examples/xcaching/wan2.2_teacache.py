from typing import Any, Dict, Optional, Tuple, Union, List
from diffusers import WanPipeline
from diffusers.models import WanTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
import torch
import numpy as np
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import argparse
import os
import torch
import pandas as pd
import random
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

from datasets import load_dataset

def teacache_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()  # batch_size * seq_len
    else:
        ts_seq_len = None

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    if ts_seq_len is not None:
        # batch_size, seq_len, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # get teacache data
    # inp = hidden_states.clone()
    # temb_ = timestep_proj.clone()
    # if temb_.ndim == 4:
    #     # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
    #     shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
    #         self.blocks[0].scale_shift_table.unsqueeze(0) + temb_.float()
    #     ).chunk(6, dim=2)
    #     # batch_size, seq_len, 1, inner_dim
    #     shift_msa = shift_msa.squeeze(2)
    #     scale_msa = scale_msa.squeeze(2)
    #     gate_msa = gate_msa.squeeze(2)
    #     c_shift_msa = c_shift_msa.squeeze(2)
    #     c_scale_msa = c_scale_msa.squeeze(2)
    #     c_gate_msa = c_gate_msa.squeeze(2)
    # else:
    #     # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
    #     shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
    #         self.blocks[0].scale_shift_table + temb_.float()
    #     ).chunk(6, dim=1)

    # norm_hidden_states = (self.blocks[0].norm1(inp.float()) * (1 + scale_msa) + shift_msa).type_as(inp)
    self.modulated_inp.append(timestep_proj.clone().to('cpu'))

    # 4. Transformer blocks
    ori_hidden_states = hidden_states.clone()
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    # residual for teacache
    previous_residual = hidden_states - ori_hidden_states
    self.pre_res.append(previous_residual.clone().to('cpu'))

    # 5. Output norm, projection & unpatchify
    if temb.ndim == 3:
        # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        # batch_size, inner_dim
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)

def get_torch_dtype(dtype_str):
    """Convert string to torch data type"""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_prompts(prompt_file, total_prompts, dataset_type='google'):
    """Load and process prompts"""
    print(f"Loading prompts from {prompt_file}...")
    
    # Choose loading method based on file extension
    if dataset_type == 'hf':
        ds = load_dataset(prompt_file, split='train')
        prompt_dataset = ds.shuffle(seed=42).select(range(total_prompts))["Prompt"]
    else:
        df = pd.read_csv(prompt_file, sep='\t', header=0)
        prompt_dataset = df['Prompt'].tolist()
        
        # Shuffle and truncate to specified number
        random.shuffle(prompt_dataset)
        prompt_dataset = prompt_dataset[:total_prompts]
    
    return {"prompt": prompt_dataset}

def get_data_diff(inp, pre_res):
    """Calculate differences for inp and pre_res"""
    modulated_inp_diff = []
    pre_res_diff = []
    
    # Process inp differences
    for k in range(1, len(inp)):
        diff = torch.abs(inp[k] - inp[k-1]).mean() / torch.abs(inp[k-1]).mean()
        diff = diff.to(torch.float32)
        modulated_inp_diff.append(diff.item())  # Convert to Python scalar
    
    # Process pre_res differences
    for k in range(1, len(pre_res)):
        diff = torch.abs(pre_res[k] - pre_res[k-1]).mean() / torch.abs(pre_res[k-1]).mean()
        diff = diff.to(torch.float32)
        pre_res_diff.append(diff.item())
    
    return modulated_inp_diff, pre_res_diff


def worker_process(gpu_id, prompts, args, start_idx):
    """Worker process for each GPU"""
    print(f"GPU {gpu_id}: Starting worker process with {len(prompts)} prompts")
    
    # Set CUDA device for this process
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    # Set torch data type
    torch_dtype = get_torch_dtype(args.torch_dtype)
    
    # Load model on specific GPU
    print(f"GPU {gpu_id}: Loading model...")
    WanTransformer3DModel.forward = teacache_forward
    pipeline = WanPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch_dtype
    )
    pipeline.to(device)
    
    all_inp_diff = []
    all_pre_res_diff = []

    # Generate images and save data
    print(f"GPU {gpu_id}: Starting image generation...")
    for local_idx, prompt in enumerate(prompts):
        global_idx = start_idx + local_idx
        print(f"GPU {gpu_id}: Processing {local_idx+1}/{len(prompts)} (global {global_idx+1}): {prompt[:50]}...")
        
        # Reset cache
        pipeline.transformer.__class__.modulated_inp = []
        pipeline.transformer.__class__.pre_res = []
        
        # Generate image
        if "5B" in args.model_path:
            # For 5B model, use 121 frames
            video = pipeline(
                prompt,
                num_inference_steps=args.num_steps,
                generator=torch.Generator("cpu").manual_seed(args.seed + global_idx),  # Different seed for each prompt
                width=args.width,
                height=args.height,
                num_frames=121,
                guidance_scale=5.0,
            )
        else:
            # For 14B model, use 61 frames
            video = pipeline(
                prompt,
                num_inference_steps=args.num_steps,
                generator=torch.Generator("cpu").manual_seed(args.seed + global_idx),  # Different seed for each prompt
                width=args.width,
                height=args.height,
                num_frames=81,
                guidance_scale=4.0,
                guidance_scale_2=3.0
            )

        # Get diff data
        inp_diff, pre_resdu_diff = get_data_diff(
            pipeline.transformer.__class__.modulated_inp, 
            pipeline.transformer.__class__.pre_res
        )
        all_inp_diff.append(inp_diff)
        all_pre_res_diff.append(pre_resdu_diff)

        # Synchronize and clear cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
    # Save results for this GPU
    output_file = f"{args.output_dir}/gpu_{gpu_id}.pth"
    print(f"GPU {gpu_id}: Saving to {output_file}")
    torch.save({
        "inp": all_inp_diff,
        "pre_res": all_pre_res_diff, 
        "gpu_id": gpu_id,
        "num_prompts": len(prompts)
    }, output_file)

    print(f"GPU {gpu_id}: Worker process completed!")
    return output_file


def split_prompts_for_gpus(prompts, num_gpus):
    """Split prompts evenly across GPUs"""
    total_prompts = len(prompts)
    prompts_per_gpu = total_prompts // num_gpus
    remainder = total_prompts % num_gpus
    
    splits = []
    start_idx = 0
    
    for i in range(num_gpus):
        # Add one extra prompt to first 'remainder' GPUs
        current_size = prompts_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + current_size
        
        splits.append({
            'prompts': prompts[start_idx:end_idx],
            'start_idx': start_idx
        })
        
        start_idx = end_idx
    
    return splits


def merge_results(output_dir, num_gpus):
    """Merge results from all GPUs"""
    print("Merging results from all GPUs...")
    
    all_inp_diff = []
    all_pre_res_diff = []
    
    for gpu_id in range(num_gpus):
        gpu_file = f"{output_dir}/gpu_{gpu_id}.pth"
        if os.path.exists(gpu_file):
            print(f"Loading results from GPU {gpu_id}...")
            data = torch.load(gpu_file, map_location='cpu')
            all_inp_diff.extend(data['inp'])
            all_pre_res_diff.extend(data['pre_res'])
        else:
            print(f"Warning: Results file for GPU {gpu_id} not found!")
    
    # Save merged results
    merged_file = f"{output_dir}/merged_results.pth"
    torch.save({
        "inp": all_inp_diff,
        "pre_res": all_pre_res_diff,
        "total_prompts": len(all_inp_diff)
    }, merged_file)
    
    print(f"Merged results saved to: {merged_file}")
    print(f"Total processed prompts: {len(all_inp_diff)}")
    
    return merged_file, all_inp_diff, all_pre_res_diff


def fit_diff_data(all_inp_diff, all_pre_res_diff):
    """Fit polynomial to the differences and plot results"""
    inp_diff_np = np.array(all_inp_diff)
    pre_res_diff_np = np.array(all_pre_res_diff)

    x = inp_diff_np.mean(axis=0)
    y = pre_res_diff_np.mean(axis=0)
    coefficients = np.polyfit(x, y, 4)
    rescale_func = np.poly1d(coefficients)
    ypred = rescale_func(x)

    plt.clf()
    plt.figure(figsize=(8,8))
    plt.plot(np.log(x), np.log(y), '*',label='log residual output diff values',color='green')
    plt.plot(np.log(x), np.log(ypred), '.',label='log polyfit values',color='blue')
    plt.xlabel(f'log input_diff')
    plt.ylabel(f'log residual_output_diff')
    plt.ylim(-3,1)
    plt.legend(loc=4) 
    plt.title('4th order My Polynomial fitting ')
    plt.tight_layout()
    plt.savefig('residual_polynomial_fitting_log.png')
    print("Polynomial coefficients:", coefficients)
    

def parse_args():
    parser = argparse.ArgumentParser(description="SD3.5 TeaCache data generation script with multi-GPU support")
    
    # Basic parameters
    parser.add_argument("--model-path", type=str, required=True,
                       default="stabilityai/stable-diffusion-3.5-medium",
                       help="Path to the sd3/sd3.5 model")
    parser.add_argument("--prompt-file", type=str, default="./google_prompts.txt",
                       help="Path to the prompt file")
    parser.add_argument("--dataset-type", type=str, default="google",
                       choices=["google", "hf"],
                       help="Type of data to load (default: google)")
    parser.add_argument("--output-dir", type=str, default=None, required=True,
                       help="Output directory")
    
    # Generation parameters
    parser.add_argument("--num-steps", type=int, default=28,
                       help="Number of inference steps")
    parser.add_argument("--width", type=int, default=2048,
                       help="Image width")
    parser.add_argument("--height", type=int, default=1024,
                       help="Image height")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Data processing parameters
    parser.add_argument("--total-prompts", type=int, default=512,
                       help="Total number of prompts to process")
    
    # Multi-GPU parameters
    parser.add_argument("--num-gpus", type=int, default=None,
                       help="Number of GPUs to use (default: all available)")
    parser.add_argument("--gpu-ids", type=str, default=None,
                       help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    
    # Other parameters
    parser.add_argument("--torch-dtype", type=str, default="bfloat16",
                       choices=["float16", "bfloat16"],
                       help="PyTorch data type")
    
    return parser.parse_args()


def main():
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    args = parse_args()
    
    # Determine GPU configuration
    available_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {available_gpus}")
    
    if args.gpu_ids:
        # Use specified GPU IDs
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        print(f"Using specified GPUs: {gpu_ids}")
    elif args.num_gpus:
        # Use first N GPUs
        gpu_ids = list(range(min(args.num_gpus, available_gpus)))
        print(f"Using first {len(gpu_ids)} GPUs: {gpu_ids}")
    else:
        # Use all available GPUs
        gpu_ids = list(range(available_gpus))
        print(f"Using all available GPUs: {gpu_ids}")
    
    if len(gpu_ids) == 0:
        raise ValueError("No GPUs available or specified!")
    
    # Create output directory
    args.output_dir = f"{args.output_dir}/{args.num_steps}-steps-{args.total_prompts}-prompts"
    os.makedirs(args.output_dir, exist_ok=True)
   
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Inference steps: {args.num_steps}")
    print(f"Image size: {args.width}x{args.height}")
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    
    # Load prompts
    prompt_dataset = load_prompts(
        args.prompt_file, 
        args.total_prompts,
        args.dataset_type
    )
    
    # Split prompts for each GPU
    prompt_splits = split_prompts_for_gpus(prompt_dataset['prompt'], len(gpu_ids))
    
    # Print split information
    for i, split in enumerate(prompt_splits):
        print(f"GPU {gpu_ids[i]}: {len(split['prompts'])} prompts (starting from index {split['start_idx']})")
    
    # Create processes for each GPU
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        split = prompt_splits[i]
        process = mp.Process(
            target=worker_process,
            args=(gpu_id, split['prompts'], args, split['start_idx'])
        )
        processes.append(process)
        process.start()
        print(f"Started process for GPU {gpu_id}")
    
    # Wait for all processes to complete
    print("Waiting for all processes to complete...")
    for i, process in enumerate(processes):
        process.join()
        print(f"GPU {gpu_ids[i]} process completed")
    
    # Merge results from all GPUs
    merged_file, all_inp_diff, all_pre_res_diff = merge_results(args.output_dir, len(gpu_ids))

    # fit diff data
    fit_diff_data(all_inp_diff, all_pre_res_diff)
    
    print("All processes completed!")
    print(f"Final merged results saved to: {merged_file}")


if __name__ == "__main__":
    main()
