import torch

import time

from diffusers import FluxPipeline

from fastdm.model_entry import FluxTransformerWrapper

model_path = "/root/pretrained-models/FLUX/FLUX.1-dev"
model_path = "/root/pretrained-models/FLUX/FLUX.1-Krea-dev"
# model_path = "/root/pretrained-models/FLUX/FLUX.1-Krea-dev/transformer"

warmup_nums = 1

#random inputs
latents = torch.rand((1,8192,64), device="cuda:0", dtype=torch.bfloat16)
timestep_ = torch.tensor([1000.0], device="cuda:0",dtype=torch.bfloat16)
guidance = torch.full([1], 3.5, device="cuda:0", dtype=torch.float32)
guidance = guidance.expand(latents.shape[0])
pooled_prompt_embeds = torch.rand((1,768), device="cuda:0", dtype=torch.bfloat16)
prompt_embeds = torch.rand((1,512,4096), device="cuda:0", dtype=torch.bfloat16)
text_ids = torch.zeros((512,3), device="cuda:0", dtype=torch.bfloat16)
latent_image_ids = torch.zeros((8192,3), device="cuda:0", dtype=torch.bfloat16)

pipe = FluxPipeline.from_pretrained(model_path, 
                                                torch_dtype=torch.bfloat16,
                                                #low_cpu_mem_usage=False,
                                                #ignore_mismatched_sizes=True
                                                )

flux = FluxTransformerWrapper(ckpt_path= pipe.transformer.state_dict(), quant_type=torch.float8_e4m3fn, kernel_backend="cuda")

#warm up
for i in range(warmup_nums):
    fake_ = flux.forward(hidden_states=latents,
                    timestep=timestep_ / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,)
torch.cuda.synchronize()

print("warm-up done, execution!")
start_ = time.time()
fake_ = flux.forward(hidden_states=latents,
                     timestep=timestep_ / 1000,
                     guidance=guidance,
                     pooled_projections=pooled_prompt_embeds,
                     encoder_hidden_states=prompt_embeds,
                     txt_ids=text_ids,
                     img_ids=latent_image_ids,
                     return_dict=False)
torch.cuda.synchronize(0)
print(f"flux-transformer-time: {(time.time()-start_)*1000}ms")