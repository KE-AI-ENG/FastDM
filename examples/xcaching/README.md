# Introduction
There are scripts for get teacache data and fit corfficient, it used in fastdm model forward.

## flux model
python flux_teacache.py --model-path FLUX/FLUX.1-dev --output-dir /teacache-temp/flux --total-prompts 2 --num-gpus 1

## sd3.5 model
python sd3.5_teacache.py --model-path stabilityai/stable-diffusion-3.5-medium --output-dir /teacache-temp/sd3.5 --total-prompts 2 --num-gpus 1

## qwenimage
python qwenimage_teacache.py --model-path Qwen/Qwen-Image --output-dir /teacache-temp/qwenimage --total-prompts 64 --num-gpus 8

## wan2.2-5B
python wan2.2_teacache.py --model-path wan/Wan2.2-TI2V-5B-Diffusers --output-dir /teacache-temp/wan2.2 --total-prompts 6 --num-gpus 6 --prompt-file Rapidata/awesome-text2video-prompts --dataset-type hf --width 1280 --height 704 --num-steps 50

## wan2.2-A14B
python wan2.2_teacache.py --model-path wan/Wan2.2-T2V-A14B-Diffusers --output-dir /teacache-temp/wan2.2-a14b --total-prompts 6 --num-gpus 6 --prompt-file Rapidata/awesome-text2video-prompts --dataset-type hf --width 1280 --height 720 --num-steps 40