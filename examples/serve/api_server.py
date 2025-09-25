#!/usr/bin/env python3
"""
基于FastDM的模型推理FastAPI服务器
支持多种架构：flux, qwen, wan等
"""

import os
import time
import json
import base64
import io
import logging
from typing import Optional, Dict, Any, List, Union
from PIL import Image
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from diffusers.utils import export_to_video
import tempfile
import uvicorn
import numpy as np
import imageio.v2 as imageio  

from fastdm.common_args import get_text_gen_parser
from fastdm.model_entry import FastDMEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="FastDM 推理服务",
    description="基于FastDM的高性能图像/视频生成API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 全局变量存储模型信息
model_info: Dict[str, Any] = {}

# Pydantic 模型定义
class GenerateRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    prompt: str = Field(..., description="生成提示词")
    negative_prompt: Optional[str] = Field(None, description="负向提示词")
    steps: Optional[int] = Field(default=25, description="采样步数")
    guidance_scale: Optional[float] = Field(default=3.5, description="引导缩放")
    true_cfg_scale: Optional[float] = Field(4.0, description="Qwen模型专用CFG缩放")
    seed: int = Field(default=0, description="随机种子，-1为随机")
    width: Optional[int] = Field(default=1024, description="图像宽度")
    height: Optional[int] = Field(default=1024, description="图像高度")
    num_frames: Optional[int] = Field(default=121, description="视频帧数（wan模型）")
    fps: int = Field(default=24, description="视频帧率（wan模型）")
    max_seq_len: Optional[int] = Field(default=512, description="最大序列长度")
    input_image: Optional[str] = Field(None, description="base64编码的源图像，仅在i2v任务中使用")

class EditRequest(GenerateRequest):
    input_images: Optional[Union[str, List[str]]] = Field(None, description="base64编码的源图像")
    blend_mode: Optional[str] = Field(
        default="list",
        description="多图处理模式: 'average' - 平均混合, 'concatenate' - 拼接, 'first' - 使用第一张, 'list' - 直接传递图片列表"
    )
    concat_direction: Optional[str] = Field(
        default="horizontal",
        description="拼接方向: 'horizontal' - 水平拼接, 'vertical' - 垂直拼接"
    )


class GenerateResponse(BaseModel):
    success: bool
    type: str  # "image" or "video"
    image: Optional[str] = None  # base64编码的图像
    video: Optional[str] = None  # base64编码的视频
    format: str
    fps: Optional[int] = None
    frames: Optional[int] = None
    generation_time: float
    model_used: str
    parameters: Optional[Dict[str, Any]] = None

class ModelInfo(BaseModel):
    model_name: str

def image_to_base64(image: Image.Image) -> str:
    """将PIL图像转换为base64字符串"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def base64_to_image(base64_str: str) -> Image.Image:
    """将base64字符串转换为PIL图像"""
    img_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(img_data))
    return image

def batch_base64_to_images(base64_list: List[str]) -> List[Image.Image]:
    """将base64字符串列表转换为PIL图像列表"""
    images = []
    for base64_str in base64_list:
        try:
            img = base64_to_image(base64_str)
            images.append(img)
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"无效的图像数据: {str(e)}")
    return images

#多图片处理函数
def process_multiple_images(images, blend_mode="list", concat_direction="horizontal"):
    """
    处理多张输入图片
    blend_mode: "average" - 平均混合, "concatenate" - 拼接, "first" - 使用第一张, "list" - 直接返回图片列表
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
        # 直接返回图片列表，不进行任何合并处理
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

@app.get("/health")
async def health() -> Response:
    """Check the health of the http server."""
    return Response(status_code=200)

@app.get("/get_model_info", response_model=ModelInfo)
async def get_model_info() -> ModelInfo:
    """获取当前模型信息"""
    return model_info

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """图像/视频生成接口"""
    log_data = request.dict()
    log_data.pop('input_images', None)  # 不记录input_images的base64数据
    logger.info(f"接收到编辑请求: {json.dumps(log_data, indent=2)}")
    # 验证参数
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="提示词不能为空")
    
    if request.model!=model_info.model_name:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {request.model}")
    
    if engine.task == 'i2v' and not request.input_image:
        raise HTTPException(status_code=400, detail="i2v任务需要提供源图像")

    # 准备生成参数
    if wan_lightning:
        request.guidance_scale = 1.0
        request.steps = 4
    generate_params = {
        'prompt': request.prompt,
        'steps': request.steps,
        'guidance_scale': request.guidance_scale,
        'gen_seed': request.seed,
        'gen_width': request.width,
        'gen_height': request.height,
        'max_seq_len': request.max_seq_len
    }
    if request.negative_prompt:
        generate_params['negative_prompt'] = request.negative_prompt

    # 为不同架构设置特定参数
    if engine.task in ['t2v', 'i2v']:
        # 视频生成
        generate_params['num_frames'] = request.num_frames 
    if engine.task == 'i2v':
        # 图像到视频生成
        try:
            input_image = base64_to_image(request.input_image)
            generate_params['src_image'] = input_image.convert("RGB")  # 确保是RGB格式
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"无效的源图像数据: {str(e)}")

    if args.architecture == 'qwen':
        # Qwen模型使用true_cfg_scale
        generate_params['true_cfg_scale'] = request.true_cfg_scale        
    
    try:
        # 执行生成
        # logger.info(f"engine generate params: {json.dumps(generate_params, indent=2)}")
        gen_start_time = time.time()
        
        output = engine.generate(**generate_params)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        generation_time = time.time() - gen_start_time
        
        logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
        
        # 处理输出
        if "wan" in args.architecture:
            # # 视频输出：保存为临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            export_to_video(output, temp_path, fps=request.fps)
            
            # 读取视频文件并编码为base64
            with open(temp_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode()
            
            # 清理临时文件
            os.unlink(temp_path)

            return GenerateResponse(
                success=True,
                type="video",
                video=video_data,
                format="mp4",
                fps=request.fps,
                frames=request.num_frames,
                generation_time=generation_time,
                model_used=request.model,
            )
            
        else:
            # 图像输出
            image_b64 = image_to_base64(output)
            
            return GenerateResponse(
                success=True,
                type="image",
                image=image_b64,
                format="png",
                generation_time=generation_time,
                model_used=request.model,
                parameters=generate_params
            )
            
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")

@app.post("/edit", response_model=GenerateResponse)
async def edit_image(request: EditRequest):
    """图像编辑接口"""
    log_data = request.dict()
    log_data.pop('input_images', None)  # 不记录input_images的base64数据
    logger.info(f"接收到编辑请求: {json.dumps(log_data, indent=2)}")
    # 验证参数
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="提示词不能为空")
    
    if request.model != model_info.model_name:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {request.model}")
    
    # 处理源图像
    input_images = []
    if isinstance(request.input_images, str):
        input_images = [request.input_images]
    else:
        input_images = request.input_images
    
    assert input_images, "输入图像不能为空"
    
    try:
        input_images = batch_base64_to_images(input_images)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无效的图像数据: {str(e)}")
    processed_images = process_multiple_images(input_images, request.blend_mode, request.concat_direction)

    # 准备生成参数
    generate_params = {
        'prompt': request.prompt,
        'steps': request.steps,
        'guidance_scale': request.guidance_scale,
        'gen_seed': request.seed,
        'gen_width': request.width,
        'gen_height': request.height,
        'max_seq_len': request.max_seq_len
    }

    # 根据blend_mode处理图片参数
    if request.blend_mode == "list" and isinstance(processed_images, list):
        # list模式：传递图片列表
        generate_params['src_image'] = [img.convert("RGB") for img in processed_images]
    else:
        # 其他模式：传递单张图片
        if processed_images:
            generate_params['src_image'] = processed_images.convert("RGB")
    
    if request.negative_prompt:
        generate_params['negative_prompt'] = request.negative_prompt
    
    try:
        # 执行编辑
        gen_start_time = time.time()
        
        output = engine.generate(
            **generate_params
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        generation_time = time.time() - gen_start_time
        
        logger.info(f"编辑完成，耗时: {generation_time:.2f}秒")
        
        # 处理输出
        image_b64 = image_to_base64(output)
        
        return GenerateResponse(
            success=True,
            type="image",
            image=image_b64,
            format="png",
            generation_time=generation_time,
            model_used=request.model,
        )
        
    except Exception as e:
        logger.error(f"编辑失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"编辑失败: {str(e)}")
if __name__ == '__main__':
    parser = get_text_gen_parser()
    # 添加FastAPI服务器参数
    parser.add_argument('--served-model-name', type=str, required=True, help='模型名称')
    parser.add_argument('--host', default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=8080, help='服务器端口')
    
    args = parser.parse_args()
        
    logger.info("🚀 启动 FastDM FastAPI 推理服务器...")

    # wan lightning
    wan_lightning = False
    if args.architecture == "wan-lightning":
        args.architecture = "wan"
        wan_lightning = True
    elif args.architecture == "wan-i2v-lightning":
        args.architecture = "wan-i2v"
        wan_lightning = True

    model_load_start = time.time()
    try:
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
        )
        model_info = ModelInfo(
            model_name=args.served_model_name
        )
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"模型加载失败: {str(e)}")
    model_load_time = time.time() - model_load_start
    logger.info(f"Model loading latency: {model_load_time:.4f} seconds")
    logger.info(f"以加载模型信息: {model_info}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )