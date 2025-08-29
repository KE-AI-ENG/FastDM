# Introduction
This project introduce how to use fastdm in comfyui.

We have implemented fastdm custom comfyui nodes in `nodes.py`.It is very easy to use. Beside, we also provide some demo workflows for different models.

# Usage

- install fastdm

    ref the [README.md](../README.md)    

- install comfyui
    ```
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI
    cp nodes.py custom_nodes/
    ```

- model path

    - sdxl: download [sdxl unet model](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/unet/diffusion_pytorch_model.fp16.safetensors) into `models/unet`
    - flux: download [flux trainsformers model](https://huggingface.co/Comfy-Org/flux1-dev/blob/main/flux1-dev.safetensors) into `models/diffusion_models`.
    - qwenimage: download [qwenimage transformers model](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/blob/main/split_files/diffusion_models/qwen_image_bf16.safetensors) into `models/diffusion_models`.
    - sd3.5: download [sd3.5 transformers model](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/transformer/diffusion_pytorch_model.safetensors) into `models/diffusion_models`.
    
- run workflow
    ```
    python main.py 
    ```

