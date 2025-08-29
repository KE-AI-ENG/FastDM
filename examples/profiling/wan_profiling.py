import torch

import time

from diffusers import WanPipeline, AutoencoderKLWan

from fastdm.model_entry import WanTransformer3DWrapper

warmup_nums = 5

#random inputs
# model_path = "/root/pretrained-models/wan/Wan2.2-T2V-A14B-Diffusers"
# latents = torch.rand((1,16,21,90,160), device="cuda:0", dtype=torch.bfloat16)
# timestep_ = torch.tensor([999], device="cuda:0",dtype=torch.int64)
# prompt_embeds = torch.rand((1,512,4096), device="cuda:0", dtype=torch.bfloat16)

model_path = "/root/pretrained-models/wan/Wan2.2-TI2V-5B-Diffusers"
latents = torch.rand((1,48,31,44,80), device="cuda:0", dtype=torch.bfloat16)
timestep_ = torch.tensor([999], device="cuda:0",dtype=torch.float32).expand((1,27280))
prompt_embeds = torch.rand((1,512,4096), device="cuda:0", dtype=torch.bfloat16)

vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=torch.bfloat16)

wan = WanTransformer3DWrapper(ckpt_path= pipe.transformer.state_dict(), quant_type=torch.float8_e4m3fn, kernel_backend="cuda", config_json=f"{model_path}/transformer/config.json")

#warm up
for i in range(warmup_nums):
    fake_ = wan.forward(hidden_states=latents,
                    timestep=timestep_,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,)
torch.cuda.synchronize()

print("warm-up done, execution!")
start_ = time.time()
fake_ = wan.forward(hidden_states=latents,
                timestep=timestep_,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,)
torch.cuda.synchronize(0)
print(f"wan-transformer-time: {(time.time()-start_)*1000}ms")