# Introduction
This a introduction about how to use the demo.

## controlnet
`controlnet_demo.py` is a demo script for using diffusers pipeline with FastDM controlnet mdoel. 

> Note: It need change some diffusers pipeline code to compatible with FastDM. ref: [fastdm controlnet pipeline](https://github.com/Lzhang-hub/diffusers/commit/cd7f4debd89793b462c797a687086ed77695cdbf)

- run `controlnet_demo.py` with command line arguments:

```bash
python controlnet_demo.py --model_type flux --model_path FLUX/FLUX.1-dev --controlnet_model  jasperai/Flux.1-dev-Controlnet-Upscaler --control_image flux_controlnet_test.png
```

- use cache
```bash
python controlnet_demo.py --model_type flux --model_path FLUX/FLUX.1-dev --controlnet_model  jasperai/Flux.1-dev-Controlnet-Upscaler --control_image flux_controlnet_test.png --cache-config ../xcaching/configs/dicache_flux.json
```