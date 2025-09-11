import os
import torch
import argparse
from safetensors.torch import load_file, save_file
from typing import Dict, Union, Optional, List
from collections import defaultdict

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='将LoRA权重合并到基础模型中，保持原始模型的safetensors文件结构'
    )
    
    # 必需参数
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='基础模型所在的目录路径'
    )
    
    parser.add_argument(
        '--lora-path',
        type=str,
        required=True,
        help='LoRA模型的safetensors文件路径'
    )
    
    # 可选参数
    parser.add_argument(
        '--target-path',
        type=str,
        default=None,
        help='合并后模型的保存目录路径，如果不指定则覆盖原始文件'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='LoRA缩放因子 (默认: 1.0)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='计算设备 (默认: cuda if available else cpu)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='是否显示详细信息'
    )
    
    return parser.parse_args()

def merge_lora_to_model(
    model_path: str,
    lora_path: str,
    alpha: float = 1.0,
    target_path: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = False
) -> bool:
    """
    将LoRA权重合并到基础模型中，保持原始模型的safetensors文件结构
    
    Args:
        model_path: 基础模型所在的目录路径
        lora_path: LoRA模型的safetensors文件路径
        alpha: LoRA缩放因子
        target_path: 合并后模型的保存目录路径，如果为None则覆盖原始文件
        device: 计算设备，默认为CPU
        verbose: 是否显示详细信息
        
    Returns:
        bool: 合并是否成功
    """
    try:
        if verbose:
            print(f"开始合并LoRA权重...")
            print(f"基础模型路径: {model_path}")
            print(f"LoRA模型路径: {lora_path}")
            print(f"使用设备: {device}")
            print(f"LoRA缩放因子: {alpha}")
        
        # 记录每个safetensors文件中包含的权重映射
        file_weight_mapping = {}
        # 加载基础模型的所有safetensors文件
        model_state_dict = {}
        
        # 首先扫描并记录每个权重属于哪个文件
        if verbose:
            print("正在加载基础模型...")
            
        for filename in os.listdir(model_path):
            if filename.endswith('.safetensors'):
                file_path = os.path.join(model_path, filename)
                if verbose:
                    print(f"加载文件: {filename}")
                state_dict = load_file(file_path, device='cpu')
                file_weight_mapping[filename] = set(state_dict.keys())
                model_state_dict.update(state_dict)
        
        # 加载LoRA权重
        if verbose:
            print("正在加载LoRA权重...")
        lora_state_dict = load_file(lora_path, device=device)
        
        # 提取LoRA权重中的up和down投影矩阵
        lora_pairs = {}
        for key in lora_state_dict:
            if 'lora_down' in key:

                down_key = key
                up_key = key.replace('lora_down', 'lora_up')

                #convert to diffusers ckpt format
                #for example: 'diffusion_model.blocks.0.cross_attn.k.lora_down.weight' to 'blocks.0.attn2.to_k.weight'
                converted_base_key = key.replace('diffusion_model.', '')
                converted_base_key = converted_base_key.replace('cross_attn', 'attn2')
                converted_base_key = converted_base_key.replace('self_attn', 'attn1')
                converted_base_key = converted_base_key.replace('k.lora_down.weight', 'to_k.weight')
                converted_base_key = converted_base_key.replace('q.lora_down.weight', 'to_q.weight')
                converted_base_key = converted_base_key.replace('v.lora_down.weight', 'to_v.weight')
                converted_base_key = converted_base_key.replace('o.lora_down.weight', 'to_out.0.weight')
                converted_base_key = converted_base_key.replace('ffn.0.lora_down.weight', 'ffn.net.0.proj.weight')
                converted_base_key = converted_base_key.replace('ffn.2.lora_down.weight', 'ffn.net.2.weight')

                if up_key in lora_state_dict:
                    lora_pairs[converted_base_key] = {
                        'down': lora_state_dict[down_key],
                        'up': lora_state_dict[up_key],
                        "alpha": lora_state_dict[down_key.replace('lora_down.weight', 'alpha')]
                    }
        
        if verbose:
            print(f"找到 {len(lora_pairs)} 个LoRA权重对")
        
        # 创建用于存储更新后权重的字典，按文件分组
        updated_weights = defaultdict(dict)
        
        # 合并LoRA权重到基础模型
        if verbose:
            print("正在合并权重...")
            
        for base_key in model_state_dict:
            # 找到这个权重属于哪个文件
            target_file = None
            for filename, weight_set in file_weight_mapping.items():
                if base_key in weight_set:
                    target_file = filename
                    break
            
            if target_file is None:
                continue
                
            # 如果这个权重需要应用LoRA
            if base_key in lora_pairs:
                if verbose:
                    print(f"正在处理权重: {base_key}")
                down = lora_pairs[base_key]['down'].to(device)
                up = lora_pairs[base_key]['up'].to(device)
                alpha_ = lora_pairs[base_key]['alpha']
                rank = down.shape[0]
                
                # 计算LoRA权重
                original_weight = model_state_dict[base_key].to(device)
                lora_weight = (up @ down) * (alpha_/rank)
                updated_weight = original_weight + lora_weight
                
                updated_weights[target_file][base_key] = updated_weight.cpu()
            else:
                # 如果不需要应用LoRA，直接复制原始权重
                updated_weights[target_file][base_key] = model_state_dict[base_key]
        
        # 保存更新后的权重，保持原始文件结构
        target_dir = target_path if target_path else model_path
        os.makedirs(target_dir, exist_ok=True)
        
        if verbose:
            print("正在保存合并后的模型...")
            
        for filename, weights in updated_weights.items():
            output_path = os.path.join(target_dir, filename)
            if verbose:
                print(f"保存文件: {output_path}")
            save_file(weights, output_path)
            
        if verbose:
            print("合并完成！")
            
        return True
        
    except Exception as e:
        print(f"合并过程中出现错误: {str(e)}")
        return False

def main():
    """
    主函数
    """
    args = parse_args()
    
    success = merge_lora_to_model(
        model_path=args.model_path,
        lora_path=args.lora_path,
        alpha=args.alpha,
        target_path=args.target_path,
        device=args.device,
        verbose=args.verbose
    )
    
    if success:
        print("LoRA权重合并成功！")
    else:
        print("LoRA权重合并失败！")
        exit(1)

if __name__ == "__main__":
    main()