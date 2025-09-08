### install dependencies:

`pip install gradio`

**注**: 直接安装gradio起服务，如果遇到`ERROR: Exception in ASGI application`的问题，可通过降低pydantic版本来解决:

`pip install pydantic==2.10.6`

### launch server:

`python gradio_launch.py --model-path /path/to/qwen-image --use-int8 --architecture qwen --device 0 --cache-config ../xcaching/configs/qwenimage.json --port 7890`

对于Qwen-Image/FLUX模型，配置`--oom-resolve`可以在4090/4090D-24GB，A100-40GB，RTX8000等小显存卡上运行避免出现OOM error(生图分辨率需要小于768)。此模式会影响生成速度，如使用较大显存的卡，不建议使能该选项。 

对于Qwen模型，小于24GB显存的卡上, 会触发更多部分的量化以进一步减少显存占用, 实测发现这会造成一些生成效果的影响。

server启动之后在浏览器中打开 http://localhost:7890 即可访问服务(如下图所示)。

**注**: 如果FastDM运行在远程服务器中，想在本地计算机浏览器中访问该网址，建议使用vscode的terminal来起服务，这样可以直接访问。否则可能需要映射ssh才能访问服务器中起的服务。

![image](../../assets/gradio-gen.PNG)

### 多图输入编辑

该服务也支持**Image-Editing**任务，在‘图片编辑设置’区域输入原始图片即可。

注意这需要载入[Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)或[FLUX-Kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)模型，使用文生图模型会报error。

图片编辑模式不建议开启`--oom-resolve`，因为这样会把image_encoder部分也在cpu运行，非常慢。

最近很火的nano-banana一种玩法就是多图编辑。FastDM也支持该模式，比如让大幂幂穿上一件粉色T恤：

`python gradio_launch.py --model-path /path/to/FLUX.1-Kontext-dev --use-int8 --architecture flux --device 0 --cache-config ../xcaching/configs/flux.json --port 7891`

![image](../../assets/multi-image.PNG)
