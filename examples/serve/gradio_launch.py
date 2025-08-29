import gc
import argparse

import numpy as np
from PIL import Image

import gradio as gr
import torch
from diffusers import DiffusionPipeline

from fastdm.model_entry import FluxTransformerWrapper, SD35TransformerWrapper, SDXLUNetModelWrapper, QwenTransformerWrapper
from fastdm.cache_config import CacheConfig

def parseArgs():
    parser = argparse.ArgumentParser(description="Options for FastDM Server", conflict_handler='resolve')
    parser.add_argument('--steps', type=int, default=25, help="Inference steps")
    parser.add_argument('--device', type=int, default=0, help="device number")

    parser.add_argument('--use-fp8', action='store_true', help="Enable fp8 model inference")
    parser.add_argument('--use-int8', action='store_true', help="Enable int8 model inference")
    parser.add_argument('--kernel-backend', default="cuda", help="kernel backend: torch/triton/cuda")

    parser.add_argument('--model-path', default='', help="Directory for diffusion model path")

    parser.add_argument('--data-type', default="bfloat16", help="data type")
    parser.add_argument('--architecture', default="flux", help="model architecture: sdxl/flux/sd3/qwen")

    # caching args
    parser.add_argument('--cache-config', type=str, default=None, help="cache config json file path")

    return parser.parse_args()

class FastDMEngine:
    def __init__(self, 
                 model_path, 
                 data_type=torch.bfloat16, 
                 device_num=0, 
                 quant_type=torch.float8_e4m3fn, 
                 kernel_backend="cuda", 
                 architecture="flux", 
                 cache_config=None):

        torch.cuda.set_device(device_num)

        self.pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=data_type, use_safetensors=True)

        if cache_config is not None:
            cache_config = CacheConfig.from_json(cache_config)
            cache_config.current_steps_callback = lambda: self.pipe.scheduler.step_index
        else:
            cache_config = None

        if "sdxl" == architecture:
           self.pipe.unet = SDXLUNetModelWrapper(self.pipe.unet.state_dict(), dtype=data_type, quant_type=quant_type, kernel_backend=kernel_backend).eval()
        elif "flux" == architecture:
            self.pipe.transformer = FluxTransformerWrapper(self.pipe.transformer.state_dict(), dtype=data_type, quant_type=quant_type, kernel_backend=kernel_backend,
                                                        cache_config=cache_config).eval()
        elif "sd3" == architecture:
            self.pipe.transformer = SD35TransformerWrapper(self.pipe.transformer.state_dict(), dtype=data_type, quant_type=quant_type, kernel_backend=kernel_backend,
                                                        cache_config=cache_config).eval()
        elif "qwen" == architecture:
            self.pipe.transformer = QwenTransformerWrapper(self.pipe.transformer.state_dict(), dtype=data_type, quant_type=quant_type, kernel_backend=kernel_backend,
                                                        cache_config=cache_config).eval()
        else:
            raise ValueError(
                f"The {architecture} model is not supported!!!"
            )

        self.pipe.to(f"cuda")

    def generate(self, prompt_text, src_image=None, gen_seed=42, inf_step=25, gen_width=2048, gen_height=1024, max_len=512, guidance=3.5):

        gen = torch.Generator().manual_seed(gen_seed)

        if src_image is None:
            images = self.pipe(prompt=prompt_text, num_inference_steps=inf_step, generator=gen, width=gen_width, height=gen_height, guidance_scale=guidance, max_sequence_length=max_len).images[0]
        else:
            images = self.pipe(prompt=prompt_text, image=src_image, num_inference_steps=inf_step, generator=gen, width=gen_width, height=gen_height, guidance_scale=guidance, max_sequence_length=max_len).images[0]
        
        return images

args = parseArgs()
torch.cuda.set_device(args.device)
engine_ = FastDMEngine(model_path=args.model_path, 
                       data_type=torch.bfloat16 if "bfloat16"==args.data_type else torch.float16, 
                       device_num=args.device, 
                       quant_type=torch.float8_e4m3fn if args.use_fp8 else (torch.int8 if args.use_int8 else None),
                       kernel_backend=args.kernel_backend, 
                       architecture=args.architecture, 
                       cache_config=args.cache_config)

# 定义图片生成函数
def generate_image_from_prompt(prompt,
                                init_image=None,
                                steps=50, 
                                cfg_scale=3.5, 
                                seed=-1,
                                sampler="Euler a",
                                width=512,
                                height=512,
                                pipeline=None):
    """
    调用推理引擎生成图片
    返回PIL图像对象或文件路径
    """

    input_image = None
    if init_image is not None and isinstance(init_image, (Image.Image, np.ndarray)):
        input_image = Image.fromarray(init_image) if isinstance(init_image, np.ndarray) else init_image
        input_image = input_image.convert("RGB")  # 确保RGB格式

    try:
        image = engine_.generate(prompt, src_image=input_image, gen_seed=seed if seed>=0 else torch.randint(0, 10000, (1,)).item(), inf_step=steps, gen_width=width, gen_height=height, guidance=cfg_scale)
        return image
    except Exception as e:
        print(f"生成失败: {e}")
        return None

# 创建带参数控制的Gradio界面
with gr.Blocks(title="FastDM生图服务", css=".gradio-container {max-width: 900px !important}") as demo:
    gr.Markdown("# 🎨 FastDM图像生成器")
    
    with gr.Row():
        with gr.Column(scale=3):
            # 主提示词输入
            prompt_input = gr.Textbox(
                label="创作提示",
                placeholder="请输入描述...",
                lines=3,
                max_lines=5
            )
            
            # 负向提示词
            neg_prompt = gr.Textbox(
                label="排除内容",
                placeholder="不希望出现的内容...",
                lines=1
            )
            
            # 图生图区域
            with gr.Accordion("🖼️ 图片编辑设置", open=False) as img2img_accordion:
                init_image = gr.Image(
                    label="输入图像",
                    type="pil",
                    height=200,
                    interactive=True
                )
                denoise_strength = gr.Slider(
                    0.01, 1.0, 
                    value=0.75, 
                    label="去噪强度",
                    info="值越高保留原始图像越少"
                )

            # 参数控制区域
            with gr.Accordion("高级参数", open=False):
                with gr.Row():
                    steps = gr.Slider(10, 100, value=50, step=1, label="迭代步数")
                    cfg_scale = gr.Slider(1.0, 20.0, value=4.0, step=0.5, label="提示权重")
                    seed = gr.Number(label="随机种子", value=-1)
                
                with gr.Row():
                    sampler = gr.Dropdown(
                        ["Euler a", "DPM++", "DDIM", "LMS"], 
                        value="Euler a", 
                        label="采样方法"
                    )
                    width = gr.Slider(256, 2048, value=512, step=64, label="宽度")
                    height = gr.Slider(256, 2048, value=512, step=64, label="高度")
            
            generate_btn = gr.Button("生成图像", variant="primary", scale=0)
        
        # 输出区域
        with gr.Column(scale=2):
            output_image = gr.Image(
                label="生成结果",
                type="pil",
                interactive=False,
                height=640, width=640
            )
            gen_info = gr.Textbox(label="参数详情", interactive=False)
    
    # 示例提示词
    gr.Examples(
        examples=[
            ["A futuristic cityscape with flying cars, neon lights, rain-soaked streets"],
            ["An astronaut riding a horse on Mars, photorealistic"],
            ["Watercolor painting of cherry blossoms falling on a japanese temple"]
        ],
        inputs=prompt_input
    )
    
    # 绑定生成事件（传入所有参数）
    inputs = [
        prompt_input,
        init_image,
        steps,
        cfg_scale,
        seed,
        sampler,
        width,
        height
    ]
    
    generate_btn.click(
        fn=generate_image_from_prompt,
        inputs=inputs,
        outputs=output_image
    )
    
    # 参数信息显示函数
    def update_gen_info(prompt, init_image, steps, cfg_scale, seed, sampler, width, height):
        info = f"""
        Steps: {steps} | CFG: {cfg_scale} | Sampler: {sampler}
        Size: {width}x{height} | Seed: {seed if seed != -1 else 'random'}
        """
        return info.strip()
    
    # 添加参数变更监听
    for component in inputs[1:]:  # 从steps开始的所有参数
        component.change(
            fn=update_gen_info,
            inputs=inputs,
            outputs=gen_info
        )

# 启动服务 (可配置参数)
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7890,
        share=False,  # 设置为True可生成临时公网链接
        show_error=True
    )