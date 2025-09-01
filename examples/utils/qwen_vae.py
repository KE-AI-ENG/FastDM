import torch

from diffusers.models.autoencoders.vae import DecoderOutput

def qwen_vae_new_encode(self, x: torch.Tensor):

    x = x.to("cuda")

    _, _, num_frame, height, width = x.shape

    if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
        return self.tiled_encode(x)

    self.clear_cache()
    iter_ = 1 + (num_frame - 1) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)

    enc = self.quant_conv(out)
    self.clear_cache()
    return enc.to("cpu")

def qwen_vae_new_decode(self, z: torch.Tensor, return_dict: bool = True):

    z = z.to("cuda")

    _, _, num_frame, height, width = z.shape
    tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
    tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

    if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
        return self.tiled_decode(z, return_dict=return_dict)

    self.clear_cache()
    x = self.post_quant_conv(z)
    for i in range(num_frame):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
        else:
            out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            out = torch.cat([out, out_], 2)

    out = torch.clamp(out, min=-1.0, max=1.0)
    self.clear_cache()
    if not return_dict:
        return (out,)

    out = out.to("cpu")

    return DecoderOutput(sample=out)