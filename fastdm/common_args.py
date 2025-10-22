import argparse

def add_common_args(parser):
    """添加所有脚本共用的基础参数"""
    
    # 模型相关参数
    parser.add_argument('--use-diffusers', action='store_true', help="Use hf-diffusers")
    parser.add_argument('--use-fp8', action='store_true', help="Enable fp8 model inference")
    parser.add_argument('--use-int8', action='store_true', help="Enable int8 model inference")
    parser.add_argument('--kernel-backend', default="cuda", help="kernel backend: torch/triton/cuda")
    parser.add_argument('--oom-resolve', action='store_true', help="resolve OOM error")
    parser.add_argument('--model-path', default='', help="Directory for diffusion model path")
    parser.add_argument('--data-type', default="bfloat16", help="data type: bfloat16/float16")
    parser.add_argument('--architecture', default="flux", help="model architecture: sdxl/flux/sd35/qwen/sdxl-controlnet/flux-controlnet")
    parser.add_argument('--cache-config', type=str, default=None, help="cache config json file path")

    # 生成参数
    parser.add_argument('--steps', type=int, default=25, help="Inference steps")
    parser.add_argument('--guidance-scale', type=float, default=3.5, help="guidance scale")
    parser.add_argument('--true-cfg-scale', type=float, default=4.0, help="true cfg scale")
    parser.add_argument('--seed', type=int, default=0, help="generation seed")
    parser.add_argument('--device', type=int, default=0, help="device number")
    parser.add_argument('--num-warmup-runs', type=int, default=0, help="Number of warmup runs")
    parser.add_argument('--prompts', type=str, default="An astronaut riding a horse", help="text prompt")
    parser.add_argument('--negative-prompts', type=str, default=None, help="negative text prompt")
    parser.add_argument('--width', type=int, default=2048, help="generation width")
    parser.add_argument('--height', type=int, default=1024, help="generation height")
    parser.add_argument('--output-path', type=str, default='output.png', help="output path")
    parser.add_argument('--task', type=str, default='t2i', choices=['t2i', 't2v', 'i2i', 'i2v'], help="task type: t2i (text-to-image), t2v (text-to-video), i2i (image-to-image), i2v (image-to-video)")

def get_text_gen_parser():
    """文本生成图像的参数解析器"""
    parser = argparse.ArgumentParser(description="Options for Text-to-Image Generation")
    add_common_args(parser)
    parser.add_argument('--max-seq-len', type=int, default=512, help="max sequence length")
    parser.add_argument('--num-frames', type=int, default=121, help="number of frames for video")
    parser.add_argument('--fps', type=int, default=24, help="FPS for video output")
    parser.add_argument('--image-path', type=str, default=None, help="input image path for img2video")
    parser.add_argument('--sparse-attn-config', type=str, default=None, help="sparse attention config json file path")
    parser.add_argument('--lora-config', type=str, default=None, help="Lora model config, is a json file path or json str, example: {'lora_name': 'path/to/lora'}")
    return parser

def get_image_edit_parser():
    """图像编辑的参数解析器"""
    parser = argparse.ArgumentParser(description="Options for Image Editing")
    add_common_args(parser)
    parser.add_argument('--image-path', type=str, required=True, help="input image path")
    return parser

def get_conrolnet_parser():
    """ControlNet的参数解析器"""
    parser = argparse.ArgumentParser(description="Options for ControlNet Generation")
    add_common_args(parser)
    parser.add_argument('--controlnet-model', type=str,required=True, help="ControlNet model path")
    parser.add_argument('--control-image-path', type=str, required=True, help="input control image path")
    parser.add_argument('--vae-model', type=str, default='madebyollin/sdxl-vae-fp16-fix', help="VAE model path")
    parser.add_argument('--controlnet-scale', type=float, default=1.0, help="ControlNet conditioning scale")
    return parser

def get_gradio_parser():
    """Gradio服务器的参数解析器"""
    parser = argparse.ArgumentParser(description="Options for FastDM Server")
    add_common_args(parser)
    parser.add_argument('--port', type=int, default=7890, help="server port")
    parser.add_argument('--share', action='store_true', help="share the gradio app")
    return parser