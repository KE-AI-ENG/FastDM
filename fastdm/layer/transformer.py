# Adapted from
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from typing import Optional, Tuple

import torch
from einops import rearrange

from fastdm.layer.qlinear import QLinear
from fastdm.layer.activations import *
from fastdm.kernel.operators_set import rms_norm, rotary_pos_embedding, scaled_dot_product_attention
from fastdm.sparse.xsparse import SparseAttn

class FeedForward:
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        data_type=torch.bfloat16
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            self.act_fn = GELU(dim, inner_dim, bias=bias, data_type=data_type)
        if activation_fn == "gelu-approximate":
            self.act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias, data_type=data_type)
        elif activation_fn == "geglu":
            self.act_fn = GEGLU(dim, inner_dim, bias=bias, data_type=data_type)
        elif activation_fn == "geglu-approximate":
            self.act_fn = ApproximateGELU(dim, inner_dim, bias=bias, data_type=data_type)
        elif activation_fn == "swiglu":
            self.act_fn = SwiGLU(dim, inner_dim, bias=bias, data_type=data_type)

        self.ff_out_proj = QLinear(inner_dim, dim_out, bias=bias, data_type=data_type)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = self.act_fn.forward(hidden_states)
        x = self.ff_out_proj.forward(x)

        return x

class Attention:
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        kv_heads (`int`,  *optional*, defaults to `None`):
            The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
            `kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
            Query Attention (MQA) otherwise GQA is used.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
        data_type = torch.bfloat16,
    ):
        super().__init__()

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        self.eps = eps

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.sdpa_kv_heads = kv_heads if kv_heads is not None else self.heads
        self.sdpa_head_dim = dim_head

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        self.group_norm = None

        self.spatial_norm = None


        self.norm_q_weight = torch.empty((dim_head), dtype = data_type)
        self.norm_k_weight = torch.empty((dim_head), dtype = data_type)

        self.norm_cross = None

        self.qkv = QLinear(query_dim, self.inner_dim + self.inner_kv_dim + self.inner_kv_dim, bias=bias, data_type=data_type)

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_qkv_proj = QLinear(added_kv_proj_dim, self.inner_kv_dim + self.inner_kv_dim + (self.inner_dim if self.context_pre_only is None else 0), bias=added_proj_bias, data_type=data_type)

        if not self.pre_only:
            self.to_out = QLinear(self.inner_dim, self.out_dim, bias=out_bias, data_type=data_type)

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = QLinear(self.inner_dim, self.out_context_dim, bias=out_bias, data_type=data_type)

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "rms_norm":
                self.norm_added_q_weight = torch.empty((dim_head), dtype = data_type)
                self.norm_added_k_weight = torch.empty((dim_head), dtype = data_type)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None, 'rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        if "image_rotary_emb" in cross_attention_kwargs:
            image_rotary_emb = cross_attention_kwargs["image_rotary_emb"]
        else:
            image_rotary_emb = None

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        qkv_fusion = self.qkv.forward(hidden_states)
        query = qkv_fusion[:, :, 0:self.inner_dim]
        key = qkv_fusion[:, :, self.inner_dim:(self.inner_dim + self.inner_kv_dim)]
        value = qkv_fusion[:, :, (self.inner_dim + self.inner_kv_dim):]

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        if self.norm_q_weight is not None:
            query = rms_norm(query.unflatten(-1, (self.heads, -1)).contiguous(), self.norm_q_weight, eps=self.eps).view(batch_size, -1, inner_dim)
        if self.norm_k_weight is not None:
            key = rms_norm(key.unflatten(-1, (self.heads, -1)).contiguous(), self.norm_k_weight, eps=self.eps).view(batch_size, -1, self.inner_kv_dim)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            # `context` projections.
            encoder_hidden_states_qkv_proj = self.add_qkv_proj.forward(encoder_hidden_states)
            encoder_hidden_states_query_proj = encoder_hidden_states_qkv_proj[:, :, 0:self.inner_dim]
            encoder_hidden_states_key_proj = encoder_hidden_states_qkv_proj[:, :, self.inner_dim:(self.inner_dim + self.inner_kv_dim)]
            encoder_hidden_states_value_proj = encoder_hidden_states_qkv_proj[:, :, (self.inner_dim + self.inner_kv_dim):]

            if self.norm_added_q_weight is not None:
                encoder_hidden_states_query_proj = rms_norm(encoder_hidden_states_query_proj.unflatten(-1, (self.heads, -1)).contiguous(), self.norm_added_q_weight, eps=self.eps).view(batch_size, -1, inner_dim)
            if self.norm_added_k_weight is not None:
                encoder_hidden_states_key_proj = rms_norm(encoder_hidden_states_key_proj.unflatten(-1, (self.heads, -1)).contiguous(), self.norm_added_k_weight, eps=self.eps).view(batch_size, -1, self.inner_kv_dim)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        if image_rotary_emb is not None:
            rotary_pos_embedding(query, key, head_dim, image_rotary_emb, is_neox=False)

        hidden_states = scaled_dot_product_attention(query, key, value, self.heads, self.sdpa_kv_heads, self.sdpa_head_dim, is_causal=self.is_causal, scale=self.scale)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            if self.context_pre_only is not None and not self.context_pre_only:
                encoder_hidden_states = self.to_add_out.forward(encoder_hidden_states)
        if not self.pre_only:
            hidden_states = self.to_out.forward(hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
        
    def forward_qwen(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ):
        if encoder_hidden_states is None:
            raise ValueError("QwenAttn forward requires encoder_hidden_states (text stream)")

        if "image_rotary_emb" in cross_attention_kwargs:
            image_rotary_emb = cross_attention_kwargs["image_rotary_emb"]
        else:
            image_rotary_emb = None

        batch_size, seq_txt, hidden_size = encoder_hidden_states.shape[0], encoder_hidden_states.shape[1], encoder_hidden_states.shape[2]
        seq_img = hidden_states.shape[1]

        head_dim = hidden_size // self.heads

        # Compute QKV for image stream (sample projections)
        img_qkv_fusion = self.qkv.forward(hidden_states)
        img_query = img_qkv_fusion[:, :, 0:self.inner_dim]
        img_key = img_qkv_fusion[:, :, self.inner_dim:(self.inner_dim + self.inner_kv_dim)]
        img_value = img_qkv_fusion[:, :, (self.inner_dim + self.inner_kv_dim):]

        # Compute QKV for text stream (context projections)
        txt_qkv_fusion = self.add_qkv_proj.forward(encoder_hidden_states)
        txt_query = txt_qkv_fusion[:, :, 0:self.inner_dim]
        txt_key = txt_qkv_fusion[:, :, self.inner_dim:(self.inner_dim + self.inner_kv_dim)]
        txt_value = txt_qkv_fusion[:, :, (self.inner_dim + self.inner_kv_dim):]

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (self.heads, -1)).contiguous()
        img_key = img_key.unflatten(-1, (self.heads, -1)).contiguous()
        img_value = img_value.unflatten(-1, (self.heads, -1)).contiguous()

        txt_query = txt_query.unflatten(-1, (self.heads, -1)).contiguous()
        txt_key = txt_key.unflatten(-1, (self.heads, -1)).contiguous()
        txt_value = txt_value.unflatten(-1, (self.heads, -1)).contiguous()

        # Apply QK normalization
        if self.norm_q_weight is not None:
            img_query = rms_norm(img_query, self.norm_q_weight, eps=self.eps)
        if self.norm_k_weight is not None:
            img_key = rms_norm(img_key, self.norm_k_weight, eps=self.eps)
        if self.norm_added_q_weight is not None:
            txt_query = rms_norm(txt_query, self.norm_added_q_weight, eps=self.eps)
        if self.norm_added_k_weight is not None:
            txt_key = rms_norm(txt_key, self.norm_added_k_weight, eps=self.eps)

        joint_query = (torch.cat([txt_query, img_query], dim=1)).view(batch_size, (seq_txt+seq_img), hidden_size)
        joint_key = (torch.cat([txt_key, img_key], dim=1)).view(batch_size, (seq_txt+seq_img), hidden_size)
        joint_value = (torch.cat([txt_value, img_value], dim=1)).view(batch_size, (seq_txt+seq_img), hidden_size)

        # Apply RoPE
        if image_rotary_emb is not None:
            rotary_pos_embedding(joint_query, joint_key, head_dim, image_rotary_emb, is_neox=False)

        # Compute joint attention
        joint_hidden_states = scaled_dot_product_attention(joint_query, joint_key, joint_value, self.heads, self.sdpa_kv_heads, self.sdpa_head_dim, is_causal=self.is_causal, scale=self.scale)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = self.to_out.forward(img_attn_output)

        txt_attn_output = self.to_add_out.forward(txt_attn_output)

        return img_attn_output, txt_attn_output
    
class WanAttention:

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        processor=None,
        is_cross_attention=None,
        is_causal: bool = False,
        data_type = torch.bfloat16,
    ):
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.is_causal = is_causal
        self.eps = eps

        self.scale = dim_head**-0.5

        self.sdpa_kv_heads = self.heads
        self.sdpa_head_dim = dim_head

        self.added_kv_proj_dim = added_kv_proj_dim

        self.norm_q_weight = torch.empty((dim_head*heads), dtype = data_type)
        self.norm_k_weight = torch.empty((dim_head*heads), dtype = data_type)

        if self.cross_attention_dim_head is not None:
            self.to_q = QLinear(dim, self.inner_dim, bias=True, data_type=data_type)
            self.to_kv = QLinear(dim, self.kv_inner_dim + self.kv_inner_dim, bias=True, data_type=data_type)
            
        else:
            self.qkv = QLinear(dim, self.inner_dim + self.kv_inner_dim + self.kv_inner_dim, bias=True, data_type=data_type)

        self.to_out = QLinear(self.inner_dim, dim, bias=True, data_type=data_type)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = QLinear(added_kv_proj_dim, self.inner_dim, bias=True, data_type=data_type)
            self.add_v_proj = QLinear(added_kv_proj_dim, self.inner_dim, bias=True, data_type=data_type)
            self.norm_added_k_weight = torch.empty((dim_head*heads), dtype = data_type)

        self.is_cross_attention = cross_attention_dim_head is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        sparse_attn: Optional[SparseAttn] = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """

        encoder_hidden_states_img = None
        if self.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if self.cross_attention_dim_head is not None:
            query = self.to_q.forward(hidden_states)
            kv_fusion = self.to_kv.forward(encoder_hidden_states)
            key = kv_fusion[:, :, 0:self.kv_inner_dim]
            value = kv_fusion[:, :, self.kv_inner_dim:]
            query = rms_norm(query, self.norm_q_weight, self.eps)
            key = rms_norm(key.contiguous(), self.norm_k_weight, self.eps)
        else:
            qkv_fusion = self.qkv.forward(hidden_states)
            query = qkv_fusion[:, :, 0:self.inner_dim]
            key = qkv_fusion[:, :, self.inner_dim:(self.inner_dim + self.kv_inner_dim)]
            value = qkv_fusion[:, :, (self.inner_dim + self.kv_inner_dim):]
            query = rms_norm(query.contiguous(), self.norm_q_weight, self.eps)
            key = rms_norm(key.contiguous(), self.norm_k_weight, self.eps)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        if rotary_emb is not None:
            #merge cos and sin to single tensor, to ultilize the custom-rope-emb-ops
            rotary_emb_merged = torch.cat((rotary_emb[0].squeeze()[:,0::2], rotary_emb[1].squeeze()[:,1::2]), dim=-1).to(hidden_states.dtype)
            rotary_pos_embedding(query, key, head_dim, rotary_emb_merged, is_neox=False)
        
        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = self.add_k_proj.forward(encoder_hidden_states_img)
            value_img = self.add_v_proj.forward(encoder_hidden_states_img)
            key_img = rms_norm(key_img, self.norm_added_k_weight, self.eps)
            hidden_states_img = scaled_dot_product_attention(query, key_img, value_img, self.heads, self.sdpa_kv_heads, self.sdpa_head_dim, is_causal=self.is_causal, scale=self.scale)
        
        if sparse_attn is not None and self.cross_attention_dim_head is None:  # sparse attention for self-attention
            current_step = sparse_attn.config.current_steps_callback() if sparse_attn.config.current_steps_callback() is not None else 0
            layer_index = kwargs.get("layer_index", 0)
            if current_step < sparse_attn.config.dense_steps or layer_index < sparse_attn.config.dense_layers:
                # use dense attention
                hidden_states = scaled_dot_product_attention(query, key, value, self.heads, self.sdpa_kv_heads, self.sdpa_head_dim, is_causal=self.is_causal, scale=self.scale)
            else:
                query = query.unflatten(2, (self.heads, -1))
                key = key.unflatten(2, (self.heads, -1))
                value = value.unflatten(2, (self.heads, -1))

                batch_size = query.shape[0]
                query = rearrange(query, "b s h d -> (b s) h d")
                key = rearrange(key, "b s h d -> (b s) h d")
                value = rearrange(value, "b s h d -> (b s) h d")
                # apply radial attention
                hidden_states = sparse_attn.apply(query=query, key=key, value=value)
                hidden_states = rearrange(hidden_states, "(b s) h d -> b s h d", b=batch_size)
                hidden_states = hidden_states.flatten(2, 3)
        else:
            hidden_states = scaled_dot_product_attention(query, key, value, self.heads, self.sdpa_kv_heads, self.sdpa_head_dim, is_causal=self.is_causal, scale=self.scale)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = self.to_out.forward(hidden_states)
        return hidden_states