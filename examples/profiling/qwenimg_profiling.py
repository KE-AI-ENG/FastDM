import torch

import time

from diffusers import QwenImagePipeline

from fastdm.model_entry import QwenTransformerWrapper

model_path = "/root/pretrained-models/Qwen/Qwen-Image"

warmup_nums = 1

#random inputs
latents = torch.rand((1,8192,64), device="cuda:0", dtype=torch.bfloat16)
timestep_ = torch.tensor([1.0], device="cuda:0",dtype=torch.bfloat16)
prompt_embeds = torch.rand((1,10,3584), device="cuda:0", dtype=torch.bfloat16)
encoder_mask = torch.ones((1,10), device="cuda:0", dtype=torch.int64)
img_shapes = [(1,64,128)]
txt_seq_lens = [10]

pipe = QwenImagePipeline.from_pretrained(model_path, 
                                                torch_dtype=torch.bfloat16,
                                                #low_cpu_mem_usage=False,
                                                #ignore_mismatched_sizes=True
                                                )

qwen_img_transformer = QwenTransformerWrapper(ckpt_path= pipe.transformer.state_dict(), quant_type=torch.int8, kernel_backend="cuda")

#warm up
for i in range(warmup_nums):
    fake_ = qwen_img_transformer.forward(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_mask=encoder_mask,
                    timestep=timestep_,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens)
torch.cuda.synchronize()

print("warm-up done, execution!")
start_ = time.time()
fake_ = qwen_img_transformer.forward(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=encoder_mask,
                timestep=timestep_,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens)
torch.cuda.synchronize(0)
print(f"qwen-image-transformer-time: {(time.time()-start_)*1000}ms")