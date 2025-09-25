import numpy as np
from PIL import Image

import gradio as gr
import torch

from fastdm.common_args import get_gradio_parser
from fastdm.model_entry import FastDMEngine

#多图片处理函数
def process_multiple_images(images, blend_mode="concatenate", concat_direction="horizontal"):
    """
    处理多张输入图片
    blend_mode: "average" - 平均混合, "concatenate" - 拼接, "first" - 使用第一张
    concat_direction: "horizontal" - 水平拼接, "vertical" - 垂直拼接
    """
    if not images or len(images) == 0:
        return None
    
    # 如果只有一张图片，直接返回
    if len(images) == 1:
        return images[0]
    
    # 转换所有图片为PIL Image格式
    pil_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            continue
        pil_images.append(pil_img.convert("RGB"))
    
    if not pil_images:
        return None
    
    if blend_mode == "first":
        return pil_images[0]
    
    elif blend_mode == "list":
        # 返回一个包含所有图片的列表
        return pil_images

    elif blend_mode == "average":
        # 统一所有图片尺寸到第一张图片的大小
        base_size = pil_images[0].size
        resized_images = [img.resize(base_size, Image.Resampling.LANCZOS) for img in pil_images]
        
        # 平均混合
        arrays = [np.array(img, dtype=np.float32) for img in resized_images]
        blended_array = np.mean(arrays, axis=0).astype(np.uint8)
        return Image.fromarray(blended_array)
    
    elif blend_mode == "concatenate":
        return concatenate_images(pil_images, concat_direction)
    
    return pil_images[0]

def concatenate_images(images, direction="horizontal"):
    """
    拼接多张图片
    direction: "horizontal" - 水平拼接, "vertical" - 垂直拼接
    """
    if not images:
        return None
    
    if len(images) == 1:
        return images[0]
    
    if direction == "horizontal":
        # 水平拼接：统一高度，宽度相加
        # 找到最小高度
        min_height = min(img.height for img in images)
        
        # 调整所有图片到相同高度，保持宽高比
        resized_images = []
        total_width = 0
        
        for img in images:
            # 计算新宽度以保持宽高比
            aspect_ratio = img.width / img.height
            new_width = int(min_height * aspect_ratio)
            resized_img = img.resize((new_width, min_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)
            total_width += new_width
        
        # 创建新画布
        concatenated = Image.new('RGB', (total_width, min_height))
        
        # 粘贴图片
        x_offset = 0
        for img in resized_images:
            concatenated.paste(img, (x_offset, 0))
            x_offset += img.width
            
        return concatenated
    
    elif direction == "vertical":
        # 垂直拼接：统一宽度，高度相加
        # 找到最小宽度
        min_width = min(img.width for img in images)
        
        # 调整所有图片到相同宽度，保持宽高比
        resized_images = []
        total_height = 0
        
        for img in images:
            # 计算新高度以保持宽高比
            aspect_ratio = img.height / img.width
            new_height = int(min_width * aspect_ratio)
            resized_img = img.resize((min_width, new_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)
            total_height += new_height
        
        # 创建新画布
        concatenated = Image.new('RGB', (min_width, total_height))
        
        # 粘贴图片
        y_offset = 0
        for img in resized_images:
            concatenated.paste(img, (0, y_offset))
            y_offset += img.height
            
        return concatenated
    
    return images[0]

args = get_gradio_parser().parse_args()
engine_ = FastDMEngine(
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
            task=args.task)

# 图片生成函数
def generate_image_from_prompt(prompt,
                                init_images=None,
                                blend_mode="concatenate",
                                concat_direction="horizontal",  # 新增拼接方向参数
                                steps=50, 
                                cfg_scale=3.5, 
                                seed=-1,
                                sampler="Euler a",
                                width=512,
                                height=512):
    """
    调用推理引擎生成图片，支持多图片输入
    返回PIL图像对象或文件路径
    """

    # Handle init_images - it could be a single image or a list of images
    if init_images is None:
        input_image = None
    else:
        # If it's a single image (not a list), convert to list
        if not isinstance(init_images, list):
            init_images = [init_images]
        
        images = [img[0] for img in init_images]

        input_image = process_multiple_images(images, blend_mode, concat_direction)
        if input_image and blend_mode!="list":
            input_image = input_image.convert("RGB")  # 确保RGB格式

    try:
        image = engine_.generate(prompt, 
                                 src_image=input_image, 
                                 gen_seed=seed if seed>=0 else torch.randint(0, 10000, (1,)).item(), 
                                 steps=steps, 
                                 gen_width=width, 
                                 gen_height=height, 
                                 guidance_scale=None if engine_.architecture == "qwen" else cfg_scale, 
                                 true_cfg_scale=cfg_scale if engine_.architecture == "qwen" else None)
        return image
    except Exception as e:
        print(f"生成失败: {e}")
        return None

# 创建带参数控制的Gradio界面
with gr.Blocks(title="FastDM生图服务", css=".gradio-container {max-width: 1200px !important}") as demo:
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
            
            # 多图生图区域
            with gr.Accordion("🖼️ 图片编辑设置(支持多图)", open=False) as img2img_accordion:
                with gr.Row():
                    with gr.Column():
                        # 使用Gallery组件支持多图片上传
                        init_images = gr.Gallery(
                            label="upload",
                            show_label=True,
                            elem_id="gallery",
                            columns=3,
                            rows=2,
                            object_fit="contain",
                            height="auto",
                            interactive=True,
                            file_types=['image'],
                            type="pil"
                        )
                
                with gr.Row():
                    # 多图混合模式选择（去掉overlay，新增concatenate）
                    blend_mode = gr.Dropdown(
                        choices=["concatenate", "average", "first", "list"],
                        value="list",
                        label="输入图片融合模式",
                        info="list: 多图组成list | concatenate: 拼接 | average: 平均混合 | first: 仅使用第一张"
                    )
                    
                    # 拼接方向选择（仅在concatenate模式下显示）
                    concat_direction = gr.Dropdown(
                        choices=["horizontal", "vertical"],
                        value="horizontal",
                        label="拼接方向",
                        info="horizontal: 水平拼接 | vertical: 垂直拼接",
                        visible=False  # 默认隐藏
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
                    steps = gr.Slider(1, 100, value=40, step=1, label="迭代步数")
                    cfg_scale = gr.Slider(1.0, 20.0, value=4.0, step=0.5, label="提示权重")
                    seed = gr.Number(label="随机种子", value=-1)
                
                with gr.Row():
                    sampler = gr.Dropdown(
                        ["Euler a", "DPM++", "DDIM", "LMS"], 
                        value="Euler a", 
                        label="采样方法"
                    )
                    width = gr.Slider(0, 2048, value=512, step=64, label="宽度")
                    height = gr.Slider(0, 2048, value=512, step=64, label="高度")
            
            generate_btn = gr.Button("生成图像", variant="primary", scale=0)
        
        # 输出区域
        with gr.Column(scale=2):
            output_image = gr.Image(
                label="生成结果",
                type="pil",
                show_label=True
                # interactive=False,
                # height=640, width=640
            )
            gen_info = gr.Textbox(label="参数详情", interactive=False)
            
            # 显示处理后的混合图片预览
            blended_preview = gr.Image(
                label="输入图片融合后预览",
                type="pil",
                interactive=False,
                height=300, width=300,
                visible=False
            )
    
    # 示例提示词
    gr.Examples(
        examples=[
            ["A futuristic cityscape with flying cars, neon lights, rain-soaked streets"],
            ["An astronaut riding a horse on Mars, photorealistic"],
            ["Watercolor painting of cherry blossoms falling on a japanese temple"]
        ],
        inputs=prompt_input
    )
    
    # 控制拼接方向选择器的显示/隐藏
    def toggle_concat_direction(blend_mode):
        if blend_mode == "concatenate":
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)
    
    blend_mode.change(
        fn=toggle_concat_direction,
        inputs=blend_mode,
        outputs=concat_direction
    )
    
    # 绑定生成事件（传入所有参数）
    inputs = [
        prompt_input,
        init_images,
        blend_mode,
        concat_direction,  # 新增拼接方向参数
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
    
    # 图片处理预览函数
    def preview_processed_image(images, blend_mode, concat_direction):
        if not images or len(images) == 0:
            return None, gr.update(visible=False)
        
        images = [img[0] for img in images]

        processed = process_multiple_images(images, blend_mode, concat_direction)
        if processed and (len(images) > 1 or blend_mode != "first") and blend_mode != "list":
            return processed, gr.update(visible=True)
        else:
            return None, gr.update(visible=False)
    
    # 当图片、处理模式或拼接方向改变时，显示预览
    init_images.change(
        fn=preview_processed_image,
        inputs=[init_images, blend_mode, concat_direction],
        outputs=[blended_preview, blended_preview]
    )
    
    blend_mode.change(
        fn=preview_processed_image,
        inputs=[init_images, blend_mode, concat_direction],
        outputs=[blended_preview, blended_preview]
    )
    
    concat_direction.change(
        fn=preview_processed_image,
        inputs=[init_images, blend_mode, concat_direction],
        outputs=[blended_preview, blended_preview]
    )
    
    # 参数信息显示函数
    def update_gen_info(prompt, init_images, blend_mode, concat_direction, steps, cfg_scale, seed, sampler, width, height):
        img_count = len(init_images) if init_images else 0
        mode_info = blend_mode
        if blend_mode == "concatenate" and img_count > 1:
            mode_info += f" ({concat_direction})"
        
        info = f"""
        Steps: {steps} | CFG: {cfg_scale} | Sampler: {sampler}
        Size: {width}x{height} | Seed: {seed if seed != -1 else 'random'}
        Input Images: {img_count} | Process Mode: {mode_info if img_count > 0 else 'N/A'}
        """
        return info.strip()
    
    # 添加参数变更监听
    for component in inputs[2:]:  # 从blend_mode开始的所有参数
        component.change(
            fn=update_gen_info,
            inputs=inputs,
            outputs=gen_info
        )

# 启动服务 (可配置参数)
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,  # 设置为True可生成临时公网链接
        show_error=True
    )