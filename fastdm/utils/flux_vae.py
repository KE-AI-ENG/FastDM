from typing import Union

import torch

from diffusers.models.autoencoders.vae import DecoderOutput

def flux_vae_new_encode(self, x: torch.Tensor):

    x = x.to("cuda")

    batch_size, num_channels, height, width = x.shape

    if self.use_tiling and (width > self.tile_sample_min_size or height > self.tile_sample_min_size):
        return self._tiled_encode(x)

    enc = self.encoder(x)
    if self.quant_conv is not None:
        enc = self.quant_conv(enc)

    return enc.to("cpu")

def flux_vae_new_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:

    z = z.to("cuda")

    if self.use_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
        return self.tiled_decode(z, return_dict=return_dict)

    if self.post_quant_conv is not None:
        z = self.post_quant_conv(z)

    dec = self.decoder(z)

    dec = dec.to("cpu")

    if not return_dict:
        return (dec,)

    return DecoderOutput(sample=dec)