qwenimage训练脚本

## 环境准备
```bash
pip install -r requirements.txt
```

## 数据准备

数据集样例：
```dataset/
├── img1.png
├── img1.txt
├── img2.jpg
├── img2.txt
├── img3.png
├── img3.txt
└── ...
```
其中img1.txt对应的是img1.png的图片描述，也就是生图的prompt。

> Tips：1、图片的描述要在保证准确完整的基础上精简；
2、在有图片的情况下，可以使用多模态大模型制作数据集。

## 训练
```bash
accelerate launch train.py --config ./configs/train_lora.yaml
```

## 推理
训练完会保存lora权重，使用fastdm推理前，需要先把lora权重merge到base model。
```bash
python lora_merge.py --model-path /path/to/Qwen-Image --lora-path /path/to/lora_weights --merged-model-path /path/to/qwen-lora-merged
```
merge lora之后就可以使用gen.py进行推理了。
