### Introduction

Fast Diffusion Models(FASTDM): A lightweight and concise implementation of Diffusion Models. It supports mainstream text-to-image/video model architectures, integrates with ComfyUI, and is compatible with various GPU architectures.

FastDM also optimizes GPU memory usage, so larger Qwen-Image series models can run smoothly even on cards with 24GB of VRam.

![image](./assets/architecture.PNG)

Please refer to [introduction](./doc/introduction.md) for more details.

### Diffusion models
There are two architectures for the diffusion model: unet and DiT, the fastdm support both of them.
#### UNET-architecthre
[StableDiffusion-XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

[SDXL-ControlNet](https://huggingface.co/collections/diffusers/sdxl-controlnets-64f9c35846f3f06f5abe351f)
#### DiT-architecthre
[FLUX](https://huggingface.co/black-forest-labs/FLUX.1-dev)/[FLUX-Krea](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev)/[FLUX-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

[QwenImage](https://huggingface.co/Qwen/Qwen-Image)/[Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)

[StableDiffusion-3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

[Wan2.2-T2V](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers)

[FLUX-Controlnet](https://huggingface.co/XLabs-AI/flux-controlnet-collections)

### Geting Started

Install fastdm from-source or via docker:

#### 1. From-source installation

##### Requirements

OS: Linux

Python: 3.9-3.12

GPU: compute capability 7.5 or higher (e.g., 4090, A100, H100, H20, RTX8000, L4 etc.).

CUDA-12.4 or later

torch 2.7 or later

##### Install

    `pip install -v -e .`

#### 2. Docker installation
- build docker
    `docker build -f Dockerfile -t fastdm:latest .`

### Usage

- use fastdm in diffusers pipeline:

    Use the gen.py script in the examples/demo folder to generate an image (for more details, refer to introduction):
    
    `python text2img_demo.py --num-warmup-runs 1 --dump-image-path ./flux-krea-fp8.png --use-fp8 --model-path /path/to/FLUX.1-Krea-dev --architecture flux --guidance-scale 4.5 --width 1024 --prompts "A frog holding a sign that says hello world"`

    **Note**: The Qwen-Image model typically requires over 50GB of vram to run, otherwise it will cause an Out-of-Memory (OOM) error. By configuring `--qwen-oom-resolve`, it can be run with only around 20GB. This allows it to run on graphics cards with low VRam, such as the A100-40G, 4090/4090D, and RTX-8000. This will cause the Transformer and VAE components to run on the GPU, while the text_encoder will run on the CPU. We recommend that the generated image resolution be less than 768x768 in this mode.

    Generating images using Python scripts is cumbersome, so we often want a convenient web UI to control image generation.
    
    You can use the gradio_launch.py ​​script in the examples/serve folder to quickly build a web service. This allows you to trigger image generation from a web browser, with flexible options for prompts and generation parameters. 
    
    For more information please refer to [gradio service demo](./examples/serve/readme.md) 

- use fastdm in comfyUI:

    Please refer to [Comfyui Usage](./comfyui//README.md)

#### LORA

Please refer to the [LORA](./examples//lora-gen/readme.md) documentation for details on how to use LORA with fastdm.

#### Image-Editing

use the `image_edit.py` script in the `examples/demo` folder to edit an image:

`python image_edit.py --model-path /path/to/Qwen-Image-Edit --use-int8 --image-path ./ast_ride_horse.png --prompts "Change the horse's color to purple, with a flash light background."`

#### Controlnet:

Please refer to readme.md in examples/demo

### Performance

text2image：

  all-models: **height = 1024，width = 2048，num_inference_steps = 25**

text2video：
    
  wan-5B: **height = 704，width = 1280，num_frames = 121，num_inference_steps = 50**
    
  wan-A14B：**height = 720，width = 1280，num_frames = 81，num_inference_steps = 40**

Note: The following data uses [SageAttention](https://github.com/thu-ml/SageAttention) for H20 performance on qwen-image and wan2.2-A14B. Other models and card types were not used. SageAttention significantly improves performance over the torch-sdpa operator. For details, refer to the [project](https://github.com/thu-ml/SageAttention). If SageAttention is installed in your environment, it will be directly called in FastDM's CUDA-backend mode.

The Qwen-Image perf-data of RTX8000 using the command `--data-type float16 --width 1024 --height 1024` to avoid OOM, the gpu-mem-usage is still in the process of optimization.

![image](./assets/perf.PNG)

### Accuracy

Please refer to [model accuracy evaluation](./examples/evaluation/README.md) for details.

Use the `clip_score.py` and `fid.py` script in the `examples/evaluation` folder to evaluate model (for more details, refer to [evaluation](./examples/evaluation/README.md)).


### Acknowledgement

We learned the design and reused code from the following projects: [Diffusers](https://github.com/huggingface/diffusers), [vLLM](https://github.com/vllm-project/vllm), [Flash-attention](https://github.com/Dao-AILab/flash-attention), [SGLang](https://github.com/sgl-project/sglang),[teacache](https://github.com/ali-vilab/TeaCache)

The cuda-backend kernels(high performance operator, [cutlass](https://github.com/NVIDIA/cutlass/tree/v4.1.0)-based gemm or self-attention-fp8) implementations adapted from vllm or sglang kernels and flash-attention. In order to clone the Cutlass source code from GitHub without using git submodule(the domestic network is often disconnected if you don't use VPN), we directly put the Cutlass header files in the csrc/include, this method is rather crude😂.