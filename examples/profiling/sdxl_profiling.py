import torch

import time

from fastdm.model_entry import SDXLUNetModelWrapper

#random inputs
sample_ = torch.empty((2,4,128,256), device="cuda:0", dtype=torch.float16)
timestep_ = torch.tensor([958.0], device="cuda:0",dtype=torch.float16)
encoder_hidden_ = torch.empty((2,77,2048), device="cuda:0", dtype=torch.float16)

model_path = "/root/pretrained-models/stabilityai/stable-diffusion-xl-base-1.0/unet/diffusion_pytorch_model.fp16.safetensors"
sdxl_unet = SDXLUNetModelWrapper(ckpt_path=model_path, quant_type=torch.float8_e4m3fn, kernel_backend="cuda")

#warm up
for i in range(5):
    fake_ = sdxl_unet.forward(sample_, timestep_, encoder_hidden_)
    #torch.cuda.empty_cache()
torch.cuda.synchronize()

print("warm-up done, excution!")
start_ = time.time()
fake_ = sdxl_unet.forward(sample_, timestep_, encoder_hidden_)
torch.cuda.synchronize(0)
print(f"unet-time: {(time.time()-start_)*1000}ms")

# from safetensors import safe_open
# with safe_open(model_path, framework="pt", device=0) as f:
#     for k in f.keys():
#         print(f"{k}:{f.get_tensor(k).shape}")