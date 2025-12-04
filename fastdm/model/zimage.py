# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_z_image.py

from typing import Tuple, Optional, List, Union
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from fastdm.model.basemodel import BaseModelCore
from fastdm.layer.embeddings import RopeEmbedder, TimestepEmbedder
from fastdm.layer.normalization import RMSNorm, LayerNorm
from fastdm.layer.transformer import Attention
from fastdm.layer.qlinear import QLinear
from fastdm.caching.xcaching import AutoCache

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

class FeedForward:
    def __init__(self, dim: int, hidden_dim: int, data_type=torch.bfloat16):
        super().__init__()
        self.w1 = QLinear(dim, hidden_dim, bias=False, data_type=data_type)
        self.w2 = QLinear(hidden_dim, dim, bias=False, data_type=data_type)
        self.w3 = QLinear(dim, hidden_dim, bias=False, data_type=data_type)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2.forward(F.silu(self.w1.forward(x)) * self.w3.forward(x))

class ZImageTransformerBlock:
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        data_type = torch.bfloat16
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads

        # Refactored to use diffusers Attention with custom processor
        # Original Z-Image params: dim, n_heads, n_kv_heads, qk_norm
        self.attention = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // n_heads,
            heads=n_heads,
            kv_heads=n_kv_heads,
            qk_norm="rms_norm" if qk_norm else None,
            eps=1e-5,
            bias=False,
            out_bias=False,
            data_type=data_type,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8), data_type=data_type)
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = QLinear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True, data_type=data_type)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation.forward(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention.forward(
                self.attention_norm1.forward(x) * scale_msa,
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + gate_msa * self.attention_norm2.forward(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2.forward(
                self.feed_forward.forward(
                    self.ffn_norm1.forward(x) * scale_mlp,
                )
            )
        else:
            # Attention block
            attn_out = self.attention.forward(
                self.attention_norm1.forward(x),
                attention_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
            x = x + self.attention_norm2.forward(attn_out)

            # FFN block
            x = x + self.ffn_norm2.forward(
                self.feed_forward.forward(
                    self.ffn_norm1.forward(x),
                )
            )

        return x

class FinalLayer:
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = QLinear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = QLinear(min(hidden_size, 256), hidden_size, bias=True)

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation.forward(torch.nn.functional.silu(c))
        x = self.norm_final.forward(x) * scale.unsqueeze(1)
        x = self.linear.forward(x)
        return x

def pad_sequence(
    sequences: Union[Tensor, list[Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
    padding_side: str = "right",
) -> Tensor:
    r"""Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length. :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including ``0``). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences`), ``T`` is the length of the longest
    sequence.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): value for padded elements. Default: ``0``.
        padding_side (str, optional): the side to pad the sequences on.
            Default: ``'right'``.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
    if not (torch.jit.is_tracing() or torch.jit.is_scripting()):
        # JIT doesn't support `Iterable`
        if not isinstance(sequences, Iterable):
            msg = (
                "pad_sequence: Expected iterable for input sequences, but got arg of type: "
                f"{type(sequences)}"
            )
            raise RuntimeError(msg)

        # In JIT context this leads to,
        # RuntimeError: cannot statically infer the expected size of a list in this context
        sequences = tuple(sequences)  # type: ignore[assignment]
    else:
        # For JIT, we only support Union[Tensor, Tuple[Tensor]]
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.unbind(0)  # type: ignore[assignment]

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    return torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value, padding_side  # type: ignore[arg-type]
    )

class ZImageTransformer2DModelCore(BaseModelCore):

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
        data_type = torch.bfloat16,
        quant_dtype: torch.dtype = torch.float8_e4m3fn,
        cache: AutoCache = None,
        oom_ressolve: bool=False, #The pipeline will running in cpu if it set to True, so we need copy tensor to gpu in forward.
    ):
        super().__init__("DiT")

        self.quant_dtype = quant_dtype

        self.need_resolve_oom = oom_ressolve

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = QLinear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True, data_type=data_type)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = all_x_embedder
        self.all_final_layer = all_final_layer
        self.noise_refiner = [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        self.context_refiner = [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        self.t_embedder = TimestepEmbedder(min(dim, 256), mid_size=1024)
        self.cap_embedder = [
            RMSNorm(cap_feat_dim, eps=norm_eps),
            QLinear(cap_feat_dim, dim, bias=True),
        ]

        self.x_pad_token = torch.empty((1, dim))
        self.cap_pad_token = torch.empty((1, dim))

        self.layers = [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        
        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        # cache
        self.cache = cache
        self.cache_config = cache.config if cache is not None else None
        self.enable_caching = self.cache_config.enable_caching if cache is not None else False

    def _pre_part_loading(self):
        self.init_weight(["all_final_layer.2-1.adaLN_modulation.1"], self.all_final_layer['2-1'].adaLN_modulation)
        self.init_weight(["all_final_layer.2-1.linear"], self.all_final_layer['2-1'].linear)

        self.init_weight(["all_x_embedder.2-1"], self.all_x_embedder['2-1'])

        self.cap_embedder[0].weight = self.init_weight(["cap_embedder.0.weight"])
        self.init_weight(["cap_embedder.1"], self.cap_embedder[1])
        self.init_weight(["t_embedder.mlp.0"], self.t_embedder.mlp[0])
        self.init_weight(["t_embedder.mlp.2"], self.t_embedder.mlp[2])

        self.cap_pad_token = self.init_weight(["cap_pad_token"])
        self.x_pad_token = self.init_weight(["x_pad_token"])

        return
    
    def _post_part_loading(self):
        return

    def _init_transformer_block_weights(self, module_list, prefix, has_modulation=False):
        """
        初始化一组 ZImageTransformerBlock 的权重
        :param module_list: 模块列表
        :param prefix: 权重名前缀，例如 'context_refiner'、'layers'、'noise_refiner'
        :param has_modulation: 是否有 adaLN_modulation
        """
        for i, block in enumerate(module_list):
            # Norm 层
            block.attention_norm1.weight = self.init_weight([f"{prefix}.{i}.attention_norm1.weight"])
            block.attention_norm2.weight = self.init_weight([f"{prefix}.{i}.attention_norm2.weight"])
            block.ffn_norm1.weight = self.init_weight([f"{prefix}.{i}.ffn_norm1.weight"])
            block.ffn_norm2.weight = self.init_weight([f"{prefix}.{i}.ffn_norm2.weight"])

            # adaLN_modulation
            if has_modulation:
                self.init_weight([f"{prefix}.{i}.adaLN_modulation.0"], block.adaLN_modulation, self.quant_dtype)

            # Attention 层
            block.attention.norm_q_weight = self.init_weight([f"{prefix}.{i}.attention.norm_q.weight"])
            block.attention.norm_k_weight = self.init_weight([f"{prefix}.{i}.attention.norm_k.weight"])
            self.init_weight(
                [f"{prefix}.{i}.attention.to_q", f"{prefix}.{i}.attention.to_k", f"{prefix}.{i}.attention.to_v"],
                block.attention.qkv,
                self.quant_dtype
            )
            self.init_weight([f"{prefix}.{i}.attention.to_out.0"], block.attention.to_out, self.quant_dtype)

            # FeedForward 层
            self.init_weight([f"{prefix}.{i}.feed_forward.w1"], block.feed_forward.w1, self.quant_dtype)
            self.init_weight([f"{prefix}.{i}.feed_forward.w2"], block.feed_forward.w2, self.quant_dtype)
            self.init_weight([f"{prefix}.{i}.feed_forward.w3"], block.feed_forward.w3, self.quant_dtype)

    def _major_parts_loading(self):
    
        self._init_transformer_block_weights(self.context_refiner, "context_refiner", has_modulation=False)
        self._init_transformer_block_weights(self.layers, "layers", has_modulation=True)
        self._init_transformer_block_weights(self.noise_refiner, "noise_refiner", has_modulation=True)
        
        return
    
    def unpatchify(self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size) -> List[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x
    
    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: List[torch.Tensor],
        all_cap_feats: List[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )
    
    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict = True
    ):

        if self.need_resolve_oom: #need copy tensor to gpu
            x = [cur_t.to(self.device) for cur_t in x]
            t = t.to(self.device)
            cap_feats = [cur_t.to(self.device) for cur_t in cap_feats]

        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder.forward(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"].forward(x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        for layer in self.noise_refiner:
            x = layer.forward(x, x_attn_mask, x_freqs_cis, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder[1].forward(self.cap_embedder[0].forward(cap_feats))
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = layer.forward(cap_feats, cap_attn_mask, cap_freqs_cis)

        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        for layer in self.layers:
            unified = layer.forward(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"].forward(unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        output = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if self.need_resolve_oom:
            output = [out_.to("cpu") for out_ in output]

        return (output,)