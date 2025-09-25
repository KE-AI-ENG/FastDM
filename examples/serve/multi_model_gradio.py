#!/usr/bin/env python3
"""
多模型Gradio服务，支持调用多个FastDM API服务
支持从多个API获取模型信息并通过页面生成图片/视频
"""

import base64
import io
import asyncio
import aiohttp
import logging
from typing import Optional, Tuple
from pydantic import BaseModel
from PIL import Image
import gradio as gr
import tempfile
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ModelInfo(BaseModel):
    model_name: str

# 配置多个API服务器地址 (可通过命令行参数覆盖)
DEFAULT_API_SERVERS = {
    "text2image": [
        "http://0.0.0.0:8083",
    ],
    "text2video": [
        "http://0.0.0.0:8085",
    ],
    "i2v": [
        "http://0.0.0.0:8084",
    ],
    "edit": [
        "http://0.0.0.0:8081",
    ]
}

# 全局变量
available_models = {"text2image": {}, "text2video": {}, "i2v": {}, "edit": {}}
model_to_api = {"text2image": {}, "text2video": {}, "i2v": {}, "edit": {}}
api_servers = DEFAULT_API_SERVERS

async def fetch_model_info_from_api(api_url: str) -> Optional[ModelInfo]:
    """从单个API获取模型信息"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{api_url}/get_model_info", timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    model_data = await response.json()
                    model_info = ModelInfo(**model_data)
                    logger.info(f"从 {api_url} 获取到模型信息: {model_info}")
                    return model_info
                else:
                    logger.warning(f"API {api_url} 返回状态码: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"无法连接到API {api_url}: {str(e)}")
        return None

async def fetch_all_models(model_type: str = None):
    """从所有配置的API服务器获取模型信息"""
    global available_models, model_to_api
    
    logger.info(f"正在获取{model_type or '所有'}API服务的模型信息...")
    
    # 根据模型类型获取对应的API服务器列表
    if model_type and model_type in api_servers:
        server_list = api_servers[model_type]
        logger.info(f"获取{model_type}模型，服务器列表: {server_list}")
    else:
        # 如果没有指定类型或类型不存在，获取所有服务器
        server_list = []
        for servers in api_servers.values():
            server_list.extend(servers)
        logger.info(f"获取所有模型，服务器列表: {server_list}")
    
    # 并发获取所有API的模型信息
    tasks = [fetch_model_info_from_api(api_url) for api_url in server_list]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 清空指定类型的模型信息
    if model_type:
        available_models[model_type].clear()
        model_to_api[model_type].clear()
    else:
        available_models = {"text2image": {}, "text2video": {}, "i2v": {}, "edit": {}}
        model_to_api = {"text2image": {}, "text2video": {}, "i2v": {}, "edit": {}}
    
    # 处理结果
    for api_url, result in zip(server_list, results):
        if result and isinstance(result, ModelInfo):
            model_info = result
            model_name = model_info.model_name
            
            # 为避免模型名冲突，使用API地址和端口作为前缀
            api_suffix = api_url.replace('http://', '').replace('https://', '').replace(':', '_').replace('/', '_')
            unique_model_name = f"{api_suffix}_{model_name}"
            
            # 确定模型属于哪个类型
            target_type = None
            if model_type:
                target_type = model_type
            else:
                # 如果没有指定类型，根据API服务器地址判断类型
                for type_name, servers in api_servers.items():
                    if api_url in servers:
                        target_type = type_name
                        break
            
            if target_type:
                available_models[target_type][unique_model_name] = {
                    'api_url': api_url,
                    'original_name': model_name,
                    'model_info': model_info
                }
                model_to_api[target_type][unique_model_name] = api_url
                logger.info(f"添加{target_type}模型: {unique_model_name} -> {api_url}")
    
    # 计算总模型数
    total_models = sum(len(models) for models in available_models.values())
    if total_models == 0:
        logger.warning("未找到任何可用的模型")
    else:
        logger.info(f"共找到 {total_models} 个可用模型")
        for type_name, models in available_models.items():
            logger.info(f"  {type_name}: {len(models)} 个模型")
    
    # 返回指定类型或所有模型的键列表
    if model_type:
        return list(available_models[model_type].keys())
    else:
        all_keys = []
        for models in available_models.values():
            all_keys.extend(models.keys())
        return all_keys

def refresh_models(model_type: str = "generate"):
    """刷新指定类型的模型列表"""
    try:
        models = asyncio.run(fetch_all_models(model_type))
        if models:
            # 创建带描述的选项列表
            choices = []
            for model_name in models:
                model_data = available_models[model_type].get(model_name, {})
                model_info = model_data.get('model_info')
                # 只显示模型名称，不显示完整的唯一标识符
                display_name = model_info.model_name if model_info else model_name
                choices.append((display_name, model_name))
            return gr.Dropdown(choices=choices, value=models[0] if models else None)
        else:
            return gr.Dropdown(choices=[("未找到可用模型", None)], value=None)
    except Exception as e:
        logger.error(f"刷新{model_type}模型列表失败: {str(e)}")
        return gr.Dropdown(choices=[("刷新失败", None)], value=None)

def refresh_text2image_models():
    """刷新文生图模型列表"""
    return refresh_models("text2image")

def refresh_text2video_models():
    """刷新文生视频模型列表"""
    return refresh_models("text2video")

def refresh_edit_models():
    """刷新编辑模型列表"""
    return refresh_models("edit")

def refresh_i2v_models():
    """刷新图生视频模型列表"""
    return refresh_models("i2v")

async def generate_edit(
    api_url: str,
    model: str,
    prompt: str,
    input_images: list,
    blend_mode: str = "first",
    concat_direction: str = "horizontal",
    negative_prompt: str = "",
    steps: int = 25,
    guidance_scale: float = 3.5,
    true_cfg_scale: float = 4.0,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024
) -> Tuple[Optional[list], str]:
    """调用API进行图片编辑"""
    
    # 获取原始模型名（移除API前缀）
    model_data = available_models["edit"].get(model, {})
    model_info = model_data.get('model_info')
    original_model_name = model_info.model_name if model_info else model_data.get('original_name', model)
    
    # 将图片转换为base64
    def image_to_base64(image_path):
        # 处理可能的tuple格式 (path, caption)
        if isinstance(image_path, tuple):
            image_path = image_path[0]
        # 确保是字符串路径
        if not isinstance(image_path, str):
            raise ValueError(f"Expected string path, got {type(image_path)}: {image_path}")
        
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    # 处理输入图片列表
    if not input_images or all(img is None for img in input_images):
        return None, "❌ 请上传至少一张输入图片"
    
    # 过滤掉None值并处理可能的tuple格式
    valid_images = []
    for img in input_images:
        if img is not None:
            if isinstance(img, tuple):
                # Gradio Gallery可能返回(path, caption)格式
                valid_images.append(img[0])
            else:
                valid_images.append(img)
    
    # 构建请求参数
    request_data = {
        "model": original_model_name,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed if seed != -1 else 0,
        "width": width,
        "height": height
    }
    logger.info(f"调用图片编辑API: {api_url}/edit, 模型: {original_model_name}, data: {request_data}")
    # 处理多张输入图片
    if len(valid_images) == 1:
        request_data["input_images"] = image_to_base64(valid_images[0])
    else:
        request_data["input_images"] = [image_to_base64(img) for img in valid_images]
        request_data["blend_mode"] = blend_mode
        if blend_mode == "concatenate":
            request_data["concat_direction"] = concat_direction
        
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/edit",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get("success"):
                        # 解码图像
                        image_data = base64.b64decode(result["image"])
                        image = Image.open(io.BytesIO(image_data))
                        
                        # 保存到临时文件
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                            image.save(tmp_file.name, 'PNG')
                            return [tmp_file.name], f"✅ 图片编辑成功! 耗时: {result.get('generation_time', 0):.2f}秒"
                    else:
                        return None, "❌ 图片编辑失败"
                else:
                    error_text = await response.text()
                    return None, f"❌ API调用失败 (状态码: {response.status}): {error_text}"
                    
    except asyncio.TimeoutError:
        return None, "❌ 请求超时，请重试"
    except Exception as e:
        logger.error(f"图片编辑失败: {str(e)}")
        return None, f"❌ 图片编辑失败: {str(e)}"

async def generate_content(
    api_url: str,
    model: str,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 25,
    guidance_scale: float = 3.5,
    true_cfg_scale: float = 4.0,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    num_frames: int = 121,
    fps: int = 24,
    input_image: str = None
) -> Tuple[Optional[list], str]:
    """调用API生成内容"""
    
    # 获取原始模型名（移除API前缀）
    # 检查模型在哪个类型中
    model_data = None
    model_type = None
    for mtype in ["text2image", "text2video", "i2v", "edit"]:
        if model in available_models[mtype]:
            model_data = available_models[mtype][model]
            model_type = mtype
            break

    if not model_data:
        # 默认使用text2image类型
        model_data = available_models["text2image"].get(model, {})
        model_type = "text2image"

    model_info = model_data.get('model_info')
    original_model_name = model_info.model_name if model_info else model_data.get('original_name', model)
    
    # 构建请求参数
    request_data = {
        "model": original_model_name,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed if seed != -1 else 0,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps
    }
    
    # 如果有输入图片，添加到请求中
    if input_image:
        def image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        request_data["input_image"] = image_to_base64(input_image)
    
    logger.info(f"调用API: {api_url}/generate, 模型: {original_model_name}, 类型: {model_type}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{api_url}/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    if result.get("success"):
                        if result.get("type") == "image":
                            # 解码图像
                            image_data = base64.b64decode(result["image"])
                            image = Image.open(io.BytesIO(image_data))
                            
                            # 保存到临时文件
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                                image.save(tmp_file.name, 'PNG')
                                return [tmp_file.name], f"✅ 图像生成成功! 耗时: {result.get('generation_time', 0):.2f}秒"
                        
                        elif result.get("type") == "video":
                            # 解码视频
                            video_data = base64.b64decode(result["video"])
                            
                            # 保存到临时文件
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                                tmp_file.write(video_data)
                                return [tmp_file.name], f"✅ 视频生成成功! 耗时: {result.get('generation_time', 0):.2f}秒, 帧数: {result.get('frames', 'N/A')}, FPS: {result.get('fps', 'N/A')}"
                    else:
                        return None, "❌ 生成失败"
                else:
                    error_text = await response.text()
                    return None, f"❌ API调用失败 (状态码: {response.status}): {error_text}"
                    
    except asyncio.TimeoutError:
        return None, "❌ 请求超时，请重试"
    except Exception as e:
        logger.error(f"生成失败: {str(e)}")
        return None, f"❌ 生成失败: {str(e)}"

def generate_edit_sync(model, prompt, input_images, blend_mode, concat_direction, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, progress=gr.Progress()):
    """图片编辑同步包装器用于Gradio"""
    if not model:
        return None, "❌ 请选择一个模型", gr.Button(interactive=True)
    
    if not prompt.strip():
        return None, "❌ 请输入提示词", gr.Button(interactive=True)
    
    # 处理输入图片（可能是单张或多张）
    logger.info(f"原始input_images类型: {type(input_images)}, 内容: {input_images}")
    
    if isinstance(input_images, str):
        # 单张图片
        input_images = [input_images]
    elif isinstance(input_images, list):
        # 多张图片，过滤掉None值并处理可能的tuple格式
        processed_images = []
        for img in input_images:
            if img is not None:
                if isinstance(img, tuple):
                    # Gradio Gallery可能返回(path, caption)格式
                    processed_images.append(img[0])
                else:
                    processed_images.append(img)
        input_images = processed_images
    elif input_images is None:
        input_images = []
    else:
        logger.error(f"意外的input_images格式: {type(input_images)}")
        return None, f"❌ 图片格式错误: {type(input_images)}", gr.Button(interactive=True)
    
    if not input_images:
        return None, "❌ 请上传至少一张输入图片", gr.Button(interactive=True)
    
    if model not in model_to_api["edit"]:
        return None, f"❌ 未找到编辑模型: {model}", gr.Button(interactive=True)
    
    api_url = model_to_api["edit"][model]
    logger.info(f"使用模型 {model} (API: {api_url}) 编辑图片，共 {len(input_images)} 张图片")
    
    # 显示正在处理状态
    progress(0, desc="🎨 正在编辑图片...")
    logger.info(f"编辑参数: prompt={prompt}, input_images={input_images}, blend_mode={blend_mode}, concat_direction={concat_direction}, steps={steps}, guidance_scale={guidance_scale}, true_cfg_scale={true_cfg_scale}, seed={seed}, width={width}, height={height}")
    try:
        result = asyncio.run(generate_edit(
            api_url, model, prompt, input_images, blend_mode, concat_direction,
            negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height
        ))
        progress(1, desc="✅ 编辑完成!")
        return result[0], result[1], gr.Button(interactive=True)
    except Exception as e:
        logger.error(f"图片编辑过程出错: {str(e)}")
        return None, f"❌ 图片编辑过程出错: {str(e)}", gr.Button(interactive=True)

def generate_text2image_sync(model, prompt, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, progress=gr.Progress()):
    """文生图同步包装器用于Gradio"""
    if not model:
        return None, "❌ 请选择一个模型", gr.Button(interactive=True)

    if not prompt.strip():
        return None, "❌ 请输入提示词", gr.Button(interactive=True)

    if model not in model_to_api["text2image"]:
        return None, f"❌ 未找到文生图模型: {model}", gr.Button(interactive=True)

    api_url = model_to_api["text2image"][model]
    logger.info(f"使用模型 {model} (API: {api_url}) 生成图片")

    # 显示正在处理状态
    progress(0, desc="🎨 正在生成图片...")

    try:
        result = asyncio.run(generate_content(
            api_url, model, prompt, negative_prompt, steps,
            guidance_scale, true_cfg_scale, seed, width, height, 1, 24, None
        ))
        progress(1, desc="✅ 图片生成完成!")
        return result[0], result[1], gr.Button(interactive=True)
    except Exception as e:
        logger.error(f"文生图过程出错: {str(e)}")
        return None, f"❌ 文生图过程出错: {str(e)}", gr.Button(interactive=True)

def generate_text2video_sync(model, prompt, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, progress=gr.Progress()):
    """文生视频同步包装器用于Gradio"""
    if not model:
        return None, "❌ 请选择一个模型", gr.Button(interactive=True)

    if not prompt.strip():
        return None, "❌ 请输入提示词", gr.Button(interactive=True)

    if model not in model_to_api["text2video"]:
        return None, f"❌ 未找到文生视频模型: {model}", gr.Button(interactive=True)

    api_url = model_to_api["text2video"][model]
    logger.info(f"使用模型 {model} (API: {api_url}) 生成视频")

    # 显示正在处理状态
    progress(0, desc="🎬 正在生成视频...")

    try:
        result = asyncio.run(generate_content(
            api_url, model, prompt, negative_prompt, steps,
            guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, None
        ))
        progress(1, desc="✅ 视频生成完成!")
        return result[0], result[1], gr.Button(interactive=True)
    except Exception as e:
        logger.error(f"文生视频过程出错: {str(e)}")
        return None, f"❌ 文生视频过程出错: {str(e)}", gr.Button(interactive=True)

def generate_i2v_sync(model, prompt, input_image, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, progress=gr.Progress()):
    """图生视频专用同步包装器用于Gradio"""
    if not model:
        return None, "❌ 请选择一个模型", gr.Button(interactive=True)

    if not prompt.strip():
        return None, "❌ 请输入提示词", gr.Button(interactive=True)

    if not input_image:
        return None, "❌ 请上传输入图片", gr.Button(interactive=True)

    if model not in model_to_api["i2v"]:
        return None, f"❌ 未找到图生视频模型: {model}", gr.Button(interactive=True)

    api_url = model_to_api["i2v"][model]
    logger.info(f"使用模型 {model} (API: {api_url}) 生成图生视频")

    # 显示正在处理状态
    progress(0, desc="🎬 正在生成视频...")

    try:
        result = asyncio.run(generate_content(
            api_url, model, prompt, negative_prompt, steps,
            guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, input_image
        ))
        progress(1, desc="✅ 视频生成完成!")
        return result[0], result[1], gr.Button(interactive=True)
    except Exception as e:
        logger.error(f"图生视频过程出错: {str(e)}")
        return None, f"❌ 图生视频过程出错: {str(e)}", gr.Button(interactive=True)


def create_gradio_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="FastDM AIGC服务", theme=gr.themes.Soft(), css="""
        /* Tab导航栏整体样式 */
        div.gradio-tabs > div.tab-nav {
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
            padding: 20px !important;
            background: linear-gradient(135deg, #f0f2ff, #e6f3ff) !important;
            border-radius: 20px !important;
            margin: 10px 0 20px 0 !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08) !important;
        }
        
        /* Tab按钮样式 */
        div.gradio-tabs > div.tab-nav > button {
            font-weight: bold !important;
            font-size: 16px !important;
            padding: 16px 32px !important;
            border-radius: 30px !important;
            margin: 0 12px !important;
            background: linear-gradient(45deg, #ffffff, #f8f9fa) !important;
            border: 2px solid #e9ecef !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            color: #495057 !important;
            min-width: 280px !important;
            text-align: center !important;
            box-shadow: 0 3px 12px rgba(0,0,0,0.12) !important;
        }
        
        /* 悬停效果 */
        div.gradio-tabs > div.tab-nav > button:hover {
            transform: translateY(-4px) scale(1.03) !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.18) !important;
            background: linear-gradient(45deg, #f8f9fa, #e9ecef) !important;
            border-color: #6c757d !important;
        }
        
        /* 选中状态 */
        div.gradio-tabs > div.tab-nav > button.selected {
            background: linear-gradient(45deg, #667eea, #764ba2) !important;
            color: white !important;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
            transform: translateY(-2px) !important;
            border-color: #667eea !important;
        }
        
        /* 通过Tab的ID来定位按钮样式 */
        /* AI创作工坊标签页 */
        div.gradio-tabs > div.tab-nav > button[data-testid*="text2image"] {
            font-size: 17px !important;
            font-weight: 900 !important;
            color: #4a5568 !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }
        
        div.gradio-tabs > div.tab-nav > button[data-testid*="text2image"]:hover {
            color: #2d3748 !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        
        div.gradio-tabs > div.tab-nav > button[data-testid*="text2image"].selected {
            background: linear-gradient(45deg, #667eea, #764ba2) !important;
            color: #ffffff !important;
            text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        }
        
        /* 图生视频标签页 */
        div.gradio-tabs > div.tab-nav > button[data-testid*="image2video"] {
            font-size: 16px !important;
            font-weight: 800 !important;
            color: #1a365d !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }
        
        div.gradio-tabs > div.tab-nav > button[data-testid*="image2video"]:hover {
            color: #2c5282 !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        
        div.gradio-tabs > div.tab-nav > button[data-testid*="image2video"].selected {
            background: linear-gradient(45deg, #4299e1, #3182ce) !important;
            color: #ffffff !important;
            text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        }
        
        /* 智能图像编辑标签页 */
        div.gradio-tabs > div.tab-nav > button[data-testid*="image_edit"] {
            font-size: 16px !important;
            font-weight: 800 !important;
            color: #702459 !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
        }
        
        div.gradio-tabs > div.tab-nav > button[data-testid*="image_edit"]:hover {
            color: #553c4a !important;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
        }
        
        div.gradio-tabs > div.tab-nav > button[data-testid*="image_edit"].selected {
            background: linear-gradient(45deg, #f093fb, #f5576c) !important;
            color: #ffffff !important;
            text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        }
    """) as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 10px;">
            <h2>🎨 FastDM AIGC服务</h2>
        </div>
        <script>
        // 动态应用Tab样式
        function applyTabStyles() {
            const tabButtons = document.querySelectorAll('.gradio-tabs .tab-nav button');
            
            if (tabButtons.length >= 3) {
                // 第一个Tab - AI创作工坊
                const firstTab = tabButtons[0];
                firstTab.style.fontSize = '17px';
                firstTab.style.fontWeight = '900';
                firstTab.style.color = '#4a5568';
                firstTab.style.textShadow = '0 1px 2px rgba(0,0,0,0.1)';
                
                // 第二个Tab - 图生视频
                const secondTab = tabButtons[1];
                secondTab.style.fontSize = '16px';
                secondTab.style.fontWeight = '800';
                secondTab.style.color = '#1a365d';
                secondTab.style.textShadow = '0 1px 2px rgba(0,0,0,0.1)';
                
                // 第三个Tab - 智能图像编辑  
                const thirdTab = tabButtons[2];
                thirdTab.style.fontSize = '16px';
                thirdTab.style.fontWeight = '800';
                thirdTab.style.color = '#702459';
                thirdTab.style.textShadow = '0 1px 2px rgba(0,0,0,0.1)';
                
                // 监听点击事件来应用选中样式
                tabButtons.forEach((button, index) => {
                    button.addEventListener('click', () => {
                        setTimeout(() => {
                            if (button.classList.contains('selected')) {
                                if (index === 0) {
                                    button.style.background = 'linear-gradient(45deg, #667eea, #764ba2)';
                                    button.style.color = '#ffffff';
                                    button.style.textShadow = '0 1px 3px rgba(0,0,0,0.3)';
                                } else if (index === 1) {
                                    button.style.background = 'linear-gradient(45deg, #4299e1, #3182ce)';
                                    button.style.color = '#ffffff';  
                                    button.style.textShadow = '0 1px 3px rgba(0,0,0,0.3)';
                                } else if (index === 2) {
                                    button.style.background = 'linear-gradient(45deg, #f093fb, #f5576c)';
                                    button.style.color = '#ffffff';  
                                    button.style.textShadow = '0 1px 3px rgba(0,0,0,0.3)';
                                }
                            }
                        }, 100);
                    });
                });
            }
        }
        
        // 页面加载后和延迟执行
        setTimeout(applyTabStyles, 1000);
        setTimeout(applyTabStyles, 2000);
        </script>
        """)
        
        with gr.Tabs():
            # 文生图标签页
            with gr.Tab(label="🎨 文生图 🖼️ | AI图片创作", id="text2image", elem_classes=["creative-tab"]):
                with gr.Row():
                    with gr.Column(scale=1.5):
                        # 模型选择
                        with gr.Group():
                            gr.Markdown("### 🤖 模型选择")
                            model_dropdown = gr.Dropdown(
                                label="选择模型",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                            refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
                        
                        # 生成参数
                        with gr.Group():
                            gr.Markdown("### ⚙️ 生成参数")
                            prompt = gr.Textbox(
                                label="提示词",
                                placeholder="输入你想生成的图片描述...",
                                lines=2
                            )
                            negative_prompt = gr.Textbox(
                                label="负向提示词 (可选)",
                                placeholder="输入不希望出现的内容...",
                                lines=1
                            )
                            
                            with gr.Row():
                                steps = gr.Slider(1, 100, 25, step=1, label="采样步数")
                                guidance_scale = gr.Slider(0.0, 20.0, 3.5, step=0.1, label="引导缩放")
                            
                            with gr.Row():
                                true_cfg_scale = gr.Slider(0.0, 20.0, 4.0, step=0.1, label="True CFG缩放 (Qwen模型)")
                                seed = gr.Number(0, label="随机种子 (-1为随机)", precision=0)
                            
                            with gr.Row():
                                width = gr.Slider(256, 2048, 768, step=64, label="宽度")
                                height = gr.Slider(256, 2048, 768, step=64, label="高度")

                        generate_btn = gr.Button("🎨 开始生成图片", variant="primary", size="lg")
                        
                    
                    with gr.Column(scale=1):
                        # 输出区域
                        with gr.Group():
                            gr.Markdown("### 🖼️ 生成结果")
                            output_gallery = gr.Gallery(
                                label="生成的图片",
                                show_label=False,
                                elem_id="gallery",
                                columns=1,
                                rows=1,
                                height=600,
                                allow_preview=True
                            )
                            status_text = gr.Textbox(
                                label="状态",
                                value="等待生成...",
                                interactive=False,
                                max_lines=2
                            )

            # 文生视频标签页
            with gr.Tab(label="🎬 文生视频 🎥 | AI视频创作", id="text2video", elem_classes=["t2v-tab"]):
                with gr.Row():
                    with gr.Column(scale=1.5):
                        # 模型选择
                        with gr.Group():
                            gr.Markdown("### 🤖 模型选择")
                            t2v_model_dropdown = gr.Dropdown(
                                label="选择模型",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                            t2v_refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")

                        # 生成参数
                        with gr.Group():
                            gr.Markdown("### ⚙️ 生成参数")
                            t2v_prompt = gr.Textbox(
                                label="提示词",
                                placeholder="描述你想要的视频内容...",
                                lines=2
                            )
                            t2v_negative_prompt = gr.Textbox(
                                label="负向提示词 (可选)",
                                placeholder="输入不希望出现的内容...",
                                lines=1
                            )

                            with gr.Row():
                                t2v_steps = gr.Slider(1, 100, 25, step=1, label="采样步数")
                                t2v_guidance_scale = gr.Slider(0.0, 20.0, 4.0, step=0.1, label="引导缩放")

                            with gr.Row():
                                t2v_true_cfg_scale = gr.Slider(0.0, 20.0, 3.0, step=0.1, label="True CFG缩放")
                                t2v_seed = gr.Number(0, label="随机种子 (-1为随机)", precision=0)

                            with gr.Row():
                                t2v_width = gr.Slider(256, 2048, 512, step=64, label="宽度")
                                t2v_height = gr.Slider(256, 2048, 512, step=64, label="高度")

                            # 视频参数
                            with gr.Row():
                                t2v_num_frames = gr.Slider(1, 300, 81, step=1, label="帧数")
                                t2v_fps = gr.Slider(1, 60, 16, step=1, label="帧率")

                        t2v_generate_btn = gr.Button("🎬 开始生成视频", variant="primary", size="lg")


                    with gr.Column(scale=1):
                        # 输出区域
                        with gr.Group():
                            gr.Markdown("### 🎥 生成结果")
                            t2v_output_gallery = gr.Gallery(
                                label="生成的视频",
                                show_label=False,
                                elem_id="t2v_gallery",
                                columns=1,
                                rows=1,
                                height=600,
                                allow_preview=True
                            )
                            t2v_status_text = gr.Textbox(
                                label="状态",
                                value="等待生成...",
                                interactive=False,
                                max_lines=2
                            )

            # 图生视频标签页
            with gr.Tab(label="🖼️ 图生视频 🚀 | 图片转视频", id="image2video", elem_classes=["i2v-tab"]):
                with gr.Row():
                    with gr.Column(scale=1.5):
                        # 模型选择
                        with gr.Group():
                            gr.Markdown("### 🤖 模型选择")
                            i2v_model_dropdown = gr.Dropdown(
                                label="选择模型",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                            i2v_refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
                        
                        # 图片上传
                        with gr.Group():
                            gr.Markdown("### 🖼️ 输入图片")
                            input_image = gr.Image(
                                label="上传图片",
                                type="filepath",
                                height=300,
                                interactive=True
                            )
                        
                        # 生成参数
                        with gr.Group():
                            gr.Markdown("### ⚙️ 生成参数")
                            i2v_prompt = gr.Textbox(
                                label="提示词",
                                placeholder="描述你想要的视频内容...",
                                lines=2
                            )
                            i2v_negative_prompt = gr.Textbox(
                                label="负向提示词 (可选)",
                                placeholder="输入不希望出现的内容...",
                                lines=1
                            )
                            
                            with gr.Row():
                                i2v_steps = gr.Slider(1, 100, 4, step=1, label="采样步数")
                                i2v_guidance_scale = gr.Slider(0.0, 20.0, 1.0, step=0.1, label="引导缩放")
                            
                            with gr.Row():
                                i2v_true_cfg_scale = gr.Slider(0.0, 20.0, 4.0, step=0.1, label="True CFG缩放")
                                i2v_seed = gr.Number(0, label="随机种子 (-1为随机)", precision=0)
                            
                            with gr.Row():
                                i2v_width = gr.Slider(256, 2048, 512, step=64, label="宽度")
                                i2v_height = gr.Slider(256, 2048, 512, step=64, label="高度")
                            
                            # 视频参数
                            with gr.Row():
                                i2v_num_frames = gr.Slider(1, 300, 81, step=1, label="帧数")
                                i2v_fps = gr.Slider(1, 60, 16, step=1, label="帧率")
                        
                        i2v_generate_btn = gr.Button("🎬 开始生成视频", variant="primary", size="lg")
                        
                    
                    with gr.Column(scale=1):
                        # 输出区域
                        with gr.Group():
                            gr.Markdown("### 🎥 生成结果")
                            i2v_output_gallery = gr.Gallery(
                                label="生成的视频", 
                                show_label=False,
                                elem_id="i2v_gallery", 
                                columns=1, 
                                rows=1, 
                                height=600,
                                allow_preview=True
                            )
                            i2v_status_text = gr.Textbox(
                                label="状态",
                                value="等待生成...",
                                interactive=False,
                                max_lines=2
                            )
            
            # 图片编辑标签页
            with gr.Tab(label="🔮 智能图像编辑 ✨ | 多图融合", id="image_edit", elem_classes=["edit-tab"]):
                with gr.Row():
                    with gr.Column(scale=1.5):
                        # 模型选择
                        with gr.Group():
                            gr.Markdown("### 🤖 模型选择")
                            edit_model_dropdown = gr.Dropdown(
                                label="选择模型",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                            edit_refresh_btn = gr.Button("🔄 刷新模型列表", variant="secondary")
                        
                        # 图片上传
                        with gr.Group():
                            gr.Markdown("### 🖼️ 图片上传")
                            input_images = gr.Gallery(
                                label="",
                                show_label=False,
                                elem_id="input_gallery",
                                columns=3,
                                rows=2,
                                height=300,
                                allow_preview=True,
                                interactive=True
                            )
                            upload_btn = gr.UploadButton(
                                "📁 上传图片",
                                file_types=["image"],
                                file_count="multiple"
                            )
                            clear_btn = gr.Button("🗑️ 清空图片", variant="secondary")
                        
                        # 多图片处理选项
                        with gr.Group():
                            gr.Markdown("### ⚙️ 多图片处理选项")
                            with gr.Row():
                                blend_mode = gr.Dropdown(
                                    choices=[
                                        ("使用第一张", "first"),
                                        ("平均混合", "average"),
                                        ("拼接", "concatenate"),
                                        ("图片列表", "list")
                                    ],
                                    value="list",
                                    label="混合模式",
                                    info="多张图片时的处理方式：list模式直接传递图片列表给模型"
                                )
                                concat_direction = gr.Dropdown(
                                    choices=[
                                        ("水平拼接", "horizontal"),
                                        ("垂直拼接", "vertical")
                                    ],
                                    value="horizontal",
                                    label="拼接方向",
                                    info="当选择拼接模式时生效",
                                    visible=True
                                )
                        
                        # 编辑参数
                        with gr.Group():
                            gr.Markdown("### ⚙️ 编辑参数")
                            edit_prompt = gr.Textbox(
                                label="编辑提示词",
                                placeholder="描述你想要的编辑效果...",
                                lines=2
                            )
                            edit_negative_prompt = gr.Textbox(
                                label="负向提示词 (可选)",
                                placeholder="输入不希望出现的内容...",
                                lines=1
                            )
                            
                            with gr.Row():
                                edit_steps = gr.Slider(1, 100, 25, step=1, label="采样步数")
                                edit_guidance_scale = gr.Slider(0.1, 20.0, 3.5, step=0.1, label="引导缩放")
                            
                            with gr.Row():
                                edit_true_cfg_scale = gr.Slider(0.1, 20.0, 4.0, step=0.1, label="True CFG缩放")
                                edit_seed = gr.Number(0, label="随机种子 (-1为随机)", precision=0)
                            
                            with gr.Row():
                                edit_width = gr.Slider(256, 2048, 1024, step=64, label="宽度")
                                edit_height = gr.Slider(256, 2048, 1024, step=64, label="高度")
                        
                        edit_generate_btn = gr.Button("✏️ 开始编辑", variant="primary", size="lg")
                        
                    
                    with gr.Column(scale=1):
                        # 输出区域
                        with gr.Group():
                            gr.Markdown("### 🖼️ 编辑结果")
                            edit_output_gallery = gr.Gallery(
                                label="编辑后的图片", 
                                show_label=False,
                                elem_id="edit_gallery", 
                                columns=1, 
                                rows=1, 
                                height=600,
                                allow_preview=True
                            )
                            edit_status_text = gr.Textbox(
                                label="状态",
                                value="等待编辑...",
                                interactive=False,
                                max_lines=2
                            )
        
        # 示例提示词
        with gr.Accordion("💡 示例提示词", open=False):
            gr.Examples(
                examples=[
                    ["一个未来科技城市的夜景，霓虹灯闪烁，飞行汽车在空中穿梭，cyberpunk风格，高清"],
                    ["宇航员在火星表面骑马，写实风格，电影质感，史诗级场景"],
                    ["樱花飞舞的日本寺庙，水彩画风格，春天的温暖阳光，宁静祥和"],
                    ["蒸汽朋克风格的机械龙，青铜和黄铜材质，精密齿轮，工业美学"],
                    ["梦幻森林中的小木屋，萤火虫飞舞，月光透过树叶，童话风格"]
                ],
                inputs=[prompt]
            )
        
        # 事件绑定 - 文生图
        refresh_btn.click(
            refresh_text2image_models,
            outputs=[model_dropdown]
        )

        # 事件绑定 - 文生视频
        t2v_refresh_btn.click(
            refresh_text2video_models,
            outputs=[t2v_model_dropdown]
        )
        
        # 事件绑定 - 图生视频
        i2v_refresh_btn.click(
            refresh_i2v_models,
            outputs=[i2v_model_dropdown]
        )
        
        # 事件绑定 - 图片编辑
        edit_refresh_btn.click(
            refresh_edit_models,
            outputs=[edit_model_dropdown]
        )
        
        # 动态显示拼接方向选项
        def update_concat_direction_visibility(blend_mode_value):
            return gr.Dropdown(visible=(blend_mode_value == "concatenate"))
        
        blend_mode.change(
            update_concat_direction_visibility,
            inputs=[blend_mode],
            outputs=[concat_direction]
        )
        
        # 图片上传和管理功能
        def handle_image_upload(files, existing_images):
            if files is None:
                return existing_images if existing_images else []
            
            # 获取当前已有的图片列表
            current_images = existing_images if existing_images else []
            
            # 添加新上传的图片
            new_files = [file.name for file in files] if isinstance(files, list) else [files.name]
            
            # 合并已有图片和新图片，避免重复
            all_images = current_images.copy()
            for new_file in new_files:
                if new_file not in all_images:
                    all_images.append(new_file)
            
            return all_images
        
        def clear_images():
            return []
        
        upload_btn.upload(
            handle_image_upload,
            inputs=[upload_btn, input_images],
            outputs=[input_images]
        )
        
        clear_btn.click(
            clear_images,
            outputs=[input_images]
        )
        
        def start_text2image_generation(*_):
            # 禁用按钮并显示队列状态
            return gr.Button(value="🕐 生成中...", interactive=False), "⏳ 文生图请求已加入队列，正在等待处理..."

        def start_text2video_generation(*_):
            # 禁用按钮并显示队列状态
            return gr.Button(value="🕐 生成中...", interactive=False), "⏳ 文生视频请求已加入队列，正在等待处理..."
        
        def start_edit(*_):
            # 禁用按钮并显示队列状态  
            return gr.Button(value="🕐 编辑中...", interactive=False), "⏳ 编辑请求已加入队列，正在等待处理..."
        
        def handle_text2image_generation(model, prompt, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, progress=gr.Progress()):
            # 调用文生图函数
            result = generate_text2image_sync(model, prompt, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, progress)
            # 返回结果和重新启用的按钮
            return result[0], result[1], gr.Button(value="🎨 开始生成图片", interactive=True)

        def handle_text2video_generation(model, prompt, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, progress=gr.Progress()):
            # 调用文生视频函数
            result = generate_text2video_sync(model, prompt, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, progress)
            # 返回结果和重新启用的按钮
            return result[0], result[1], gr.Button(value="🎬 开始生成视频", interactive=True)
        
        def handle_edit(model, prompt, input_images, blend_mode, concat_direction, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, progress=gr.Progress()):
            # 调用实际编辑函数
            result = generate_edit_sync(model, prompt, input_images, blend_mode, concat_direction, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, progress)
            # 返回结果和重新启用的按钮
            return result[0], result[1], gr.Button(value="✏️ 开始编辑", interactive=True)

        def handle_i2v_generation(model, prompt, input_image, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, progress=gr.Progress()):
            # 调用图生视频专用函数，传入输入图片
            result = generate_i2v_sync(model, prompt, input_image, negative_prompt, steps, guidance_scale, true_cfg_scale, seed, width, height, num_frames, fps, progress)
            # 返回结果和重新启用的按钮
            return result[0], result[1], gr.Button(value="🎬 开始生成视频", interactive=True)
        
        def start_i2v_generation(*_):
            # 禁用按钮并显示队列状态
            return gr.Button(value="🕐 生成中...", interactive=False), "⏳ 图生视频请求已加入队列，正在等待处理..."
        
        # 文生图 - 点击时先禁用按钮
        generate_btn.click(
            start_text2image_generation,
            inputs=[
                model_dropdown, prompt, negative_prompt, steps, guidance_scale,
                true_cfg_scale, seed, width, height
            ],
            outputs=[generate_btn, status_text],
            queue=False  # 立即执行按钮状态更新
        ).then(
            handle_text2image_generation,
            inputs=[
                model_dropdown, prompt, negative_prompt, steps, guidance_scale,
                true_cfg_scale, seed, width, height
            ],
            outputs=[output_gallery, status_text, generate_btn],
            queue=True  # 排队处理实际生成
        )

        # 文生视频 - 点击时先禁用按钮
        t2v_generate_btn.click(
            start_text2video_generation,
            inputs=[
                t2v_model_dropdown, t2v_prompt, t2v_negative_prompt, t2v_steps, t2v_guidance_scale,
                t2v_true_cfg_scale, t2v_seed, t2v_width, t2v_height, t2v_num_frames, t2v_fps
            ],
            outputs=[t2v_generate_btn, t2v_status_text],
            queue=False  # 立即执行按钮状态更新
        ).then(
            handle_text2video_generation,
            inputs=[
                t2v_model_dropdown, t2v_prompt, t2v_negative_prompt, t2v_steps, t2v_guidance_scale,
                t2v_true_cfg_scale, t2v_seed, t2v_width, t2v_height, t2v_num_frames, t2v_fps
            ],
            outputs=[t2v_output_gallery, t2v_status_text, t2v_generate_btn],
            queue=True  # 排队处理实际生成
        )
        
        # 图生视频 - 点击时先禁用按钮
        i2v_generate_btn.click(
            start_i2v_generation,
            inputs=[
                i2v_model_dropdown, i2v_prompt, input_image, i2v_negative_prompt, i2v_steps, i2v_guidance_scale,
                i2v_true_cfg_scale, i2v_seed, i2v_width, i2v_height, i2v_num_frames, i2v_fps
            ],
            outputs=[i2v_generate_btn, i2v_status_text],
            queue=False  # 立即执行按钮状态更新
        ).then(
            handle_i2v_generation,
            inputs=[
                i2v_model_dropdown, i2v_prompt, input_image, i2v_negative_prompt, i2v_steps, i2v_guidance_scale,
                i2v_true_cfg_scale, i2v_seed, i2v_width, i2v_height, i2v_num_frames, i2v_fps
            ],
            outputs=[i2v_output_gallery, i2v_status_text, i2v_generate_btn],
            queue=True  # 排队处理实际生成
        )
        
        # 图片编辑 - 点击时先禁用按钮
        edit_generate_btn.click(
            start_edit,
            inputs=[
                edit_model_dropdown, edit_prompt, input_images, blend_mode, concat_direction,
                edit_negative_prompt, edit_steps, edit_guidance_scale, edit_true_cfg_scale, edit_seed, edit_width, edit_height
            ],
            outputs=[edit_generate_btn, edit_status_text],
            queue=False  # 立即执行按钮状态更新
        ).then(
            handle_edit,
            inputs=[
                edit_model_dropdown, edit_prompt, input_images, blend_mode, concat_direction,
                edit_negative_prompt, edit_steps, edit_guidance_scale, edit_true_cfg_scale, edit_seed, edit_width, edit_height
            ],
            outputs=[edit_output_gallery, edit_status_text, edit_generate_btn],
            queue=True  # 排队处理实际编辑
        )
        
        # 设置队列，支持排队和进度显示
        demo.queue(
            max_size=10,           # 最大队列长度
            default_concurrency_limit=1  # 同时处理1个请求
        )

        # 启动时刷新模型列表
        demo.load(
            refresh_text2image_models,
            outputs=[model_dropdown]
        )
        demo.load(
            refresh_text2video_models,
            outputs=[t2v_model_dropdown]
        )
        demo.load(
            refresh_i2v_models,
            outputs=[i2v_model_dropdown]
        )
        demo.load(
            refresh_edit_models,
            outputs=[edit_model_dropdown]
        )
    
    return demo

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FastDM多模型Gradio服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="是否创建公共链接")
    parser.add_argument("--t2i-servers", nargs="+", default=DEFAULT_API_SERVERS["text2image"],
                       help="文生图模型API服务器地址列表")
    parser.add_argument("--t2v-servers", nargs="+", default=DEFAULT_API_SERVERS["text2video"],
                       help="文生视频模型API服务器地址列表")
    parser.add_argument("--i2v-servers", nargs="+", default=DEFAULT_API_SERVERS["i2v"],
                       help="图生视频模型API服务器地址列表")
    parser.add_argument("--edit-servers", nargs="+", default=DEFAULT_API_SERVERS["edit"],
                       help="编辑模型API服务器地址列表")
    
    args = parser.parse_args()
    
    # 更新全局API服务器列表
    global api_servers
    api_servers = {
        "text2image": args.t2i_servers,
        "text2video": args.t2v_servers,
        "i2v": args.i2v_servers,
        "edit": args.edit_servers
    }
    
    logger.info("🚀 启动 FastDM 多模型 Gradio 服务...")
    logger.info(f"📍 访问地址: http://{args.host}:{args.port}")
    logger.info(f"🔗 配置的API服务器: {api_servers}")
    
    # 创建界面
    demo = create_gradio_interface()
    
    # 启动服务
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        max_threads=10,
    )

if __name__ == "__main__":
    main()