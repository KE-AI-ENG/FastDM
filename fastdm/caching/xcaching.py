import torch
import torch.nn.functional as F
from typing import Optional, Type, Dict, Any
import numpy as np

from fastdm.caching.config import CacheConfig, TeaCacheConfig, FBCacheConfig, DiCacheConfig
from fastdm.sparse.xsparse import SparseAttn

class AutoCache:
    _registry: Dict[str, Type["AutoCache"]] = {}

    def __init__(self, config: CacheConfig):
        self.config = config
        self.accumulated_rel_l1_distance_dict = {
            "positive": 0.0,
            "negative": 0.0
        }
        self.previous_modulated_input_dict = {
            "positive": None,
            "negative": None
        }
        self.previous_residual_dict = {
            "positive": None,
            "negative": None
        }
        self.previous_encoder_residual_dict = {
            "positive": None,
            "negative": None
        }
        self.cache_status = {
            "positive": True,
            "negative": False
        }

    @classmethod
    def register(cls, name: str):
        def decorator(sub_cls):
            cls._registry[name.lower()] = sub_cls
            return sub_cls
        return decorator
    

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoCache":
        """Create a cache instance from a dictionary."""
        config = CacheConfig.from_dict(data)
        algo = config.cache_algorithm.lower()
        cache_cls = cls._registry.get(algo)
        if cache_cls is None:
            raise ValueError(f"Unknown cache algorithm: {algo}")
        return cache_cls(config)

    @classmethod
    def from_json(cls, file_path: str) -> "AutoCache":
        """Load cache configuration from a JSON file and create a cache instance."""
        config = CacheConfig.from_json(file_path)
        algo = config.cache_algorithm.lower()
        cache_cls = cls._registry.get(algo)
        if cache_cls is None:
            raise ValueError(f"Unknown cache algorithm: {algo}")
        return cache_cls(config)

    def get_current_step(self):
        return self.config.current_steps_callback() if self.config.current_steps_callback() is not None else 0
    
    def get_cache_key(self):
        """Get cache key and update cache status"""
        if self.config.negtive_cache:
            cache_key = None
            for k in self.cache_status:
                if self.cache_status[k] and cache_key is None:
                    cache_key = k
                self.cache_status[k] = not self.cache_status[k]
        else:
            cache_key = "positive"
        return cache_key
    
    def _process_controlnet_residual(self, hidden_states, controlnet_samples, total_blocks, index_block, blocks_repeat=False):
        """Common ControlNet residual processing"""
        if controlnet_samples is not None:
            interval_control = len(total_blocks) / len(controlnet_samples)
            interval_control = int(np.ceil(interval_control))
            if blocks_repeat:
                hidden_states = (
                    hidden_states + controlnet_samples[index_block % len(controlnet_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_samples[index_block // interval_control]
        return hidden_states

    def _forward_wan_blocks(self, blocks, hidden_states, encoder_hidden_states, temb, image_rotary_emb, sparse_attn):
        """Forward processing for WAN model type"""
        for layer_index, block in enumerate(blocks):
            layer_index = layer_index + 1  # Adjust layer index for WAN
            hidden_states = block.forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb, sparse_attn, layer_index)
        return hidden_states
    
    def _forward_transformer_blocks(self, model_type, blocks, hidden_states, encoder_hidden_states=None, 
                                  encoder_hidden_states_mask=None, temb=None, image_rotary_emb=None, 
                                  attention_kwargs=None, controlnet_block_samples=None, 
                                  controlnet_blocks_repeat=False, start_idx=0):
        """Common transformer blocks forward processing for complex model types"""
        if model_type not in ["qwenimage", "flux", "sd35"]:
            raise ValueError(f"Unsupported model type for _forward_transformer_blocks: {model_type}")
            
        for index_block, block in enumerate(blocks):
            if model_type == "qwenimage":
                encoder_hidden_states, hidden_states = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
            elif model_type == "flux":
                encoder_hidden_states, hidden_states = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
                hidden_states = self._process_controlnet_residual(
                    hidden_states, controlnet_block_samples, blocks, 
                    index_block + start_idx, controlnet_blocks_repeat
                )
            elif model_type == "sd35":
                encoder_hidden_states, hidden_states = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=attention_kwargs,
                )
        
        return encoder_hidden_states, hidden_states

    def apply_cache(
        self,
        model_type: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        transformer_blocks: Optional[torch.nn.ModuleList] = None,
        single_transformer_blocks: Optional[torch.nn.ModuleList] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples = True,
        controlnet_blocks_repeat: bool = False,
        sparse_attn: Optional[SparseAttn] = None
    ):
        raise NotImplementedError("Subclasses must implement apply_cache method")

@AutoCache.register("teacache")
class TeaCache(AutoCache):
    def __init__(self, config: TeaCacheConfig):
        super().__init__(config)
        self.coefficients = {
            "positive": config.coefficients,
            "negative": config.negtive_coefficients
        }

    def _get_modulated_input(self, model_type, hidden_states, encoder_hidden_states, temb, transformer_blocks):
        """Get modulated input for TeaCache"""
        inp = hidden_states.clone()
        inp1 = encoder_hidden_states.clone()
        temb_ = temb.clone()

        if model_type == "qwenimage":
            img_mod_params = transformer_blocks[0].img_mod_proj.forward(F.silu(temb_))
            txt_mod_params = transformer_blocks[0].txt_mod_proj.forward(F.silu(temb_))
            img_mod1, _ = img_mod_params.chunk(2, dim=-1)
            txt_mod1, _ = txt_mod_params.chunk(2, dim=-1)
            img_normed = F.layer_norm(inp, (transformer_blocks[0].dim,), eps=transformer_blocks[0].eps)
            txt_normed = F.layer_norm(inp1, (transformer_blocks[0].dim,), eps=transformer_blocks[0].eps)
            _, _ = transformer_blocks[0]._modulate(img_normed, img_mod1)
            txt_modulated, _ = transformer_blocks[0]._modulate(txt_normed, txt_mod1)
            return txt_modulated.clone()
        elif model_type in ["flux", "sd35"]:
            modulated_inp, *_ = transformer_blocks[0].norm1.forward(inp, emb=temb_)
            return modulated_inp
        else:
            raise ValueError(f"TeaCache unsupported model type: {model_type}. Supported types: qwenimage, flux, sd35")

    def apply_cache(
        self,
        model_type: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        transformer_blocks: Optional[torch.nn.ModuleList] = None,
        single_transformer_blocks: Optional[torch.nn.ModuleList] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples = True,
        controlnet_blocks_repeat: bool = False,
        sparse_attn: Optional[SparseAttn] = None
    ):
        current_step = self.get_current_step()
        modulated_inp = self._get_modulated_input(model_type, hidden_states, encoder_hidden_states, temb, transformer_blocks)
        cache_key = self.get_cache_key()

        # judge whether to use cache
        if current_step ==0 :
            should_calc = True
            self.accumulated_rel_l1_distance_dict[cache_key] = 0
        else: 
            coefficients = self.coefficients[cache_key]
            rescale_func = np.poly1d(coefficients)
            rel_distance = ((modulated_inp - self.previous_modulated_input_dict[cache_key]).abs().mean() / 
                          self.previous_modulated_input_dict[cache_key].abs().mean()).cpu().item()
            self.accumulated_rel_l1_distance_dict[cache_key] += rescale_func(rel_distance)
            
            if self.accumulated_rel_l1_distance_dict[cache_key] < self.config.threshold:
                print(f"Skipping calculation at step {current_step} with accumulated relative L1 distance: {self.accumulated_rel_l1_distance_dict[cache_key]:.4f}")
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance_dict[cache_key] = 0
                
        self.previous_modulated_input_dict[cache_key] = modulated_inp

        if not should_calc:
            hidden_states += self.previous_residual_dict[cache_key]
        else:
            ori_hidden_states = hidden_states.clone()
            
            if model_type == "qwenimage":
                encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                    model_type, transformer_blocks, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs
                )
            elif model_type == "flux":
                encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                    model_type, transformer_blocks, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs,
                    controlnet_block_samples, controlnet_blocks_repeat
                )
                
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                
                for index_block, block in enumerate(single_transformer_blocks):
                    hidden_states = block.forward(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                    if controlnet_single_block_samples is not None:
                        interval_control = len(single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )
                        
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
            elif model_type == "sd35":
                encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                    model_type, transformer_blocks, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs
                )
                
                if controlnet_block_samples is not None:
                    interval_control = len(transformer_blocks) // len(controlnet_block_samples)
                    hidden_states = hidden_states + controlnet_block_samples[-1 // interval_control]
            else:
                raise ValueError(f"TeaCache unsupported model type: {model_type}")

            self.previous_residual_dict[cache_key] = hidden_states - ori_hidden_states
            
        return hidden_states


@AutoCache.register("fbcache")
class FBCache(AutoCache):
    def __init__(self, config: FBCacheConfig):
        super().__init__(config)

    def _process_first_block(self, model_type, hidden_states, encoder_hidden_states, 
                           encoder_hidden_states_mask, temb, image_rotary_emb, 
                           attention_kwargs, transformer_blocks, controlnet_block_samples):
        """Process first transformer block for FBCache"""
        if model_type == "wan":
            first_hidden_states = transformer_blocks[0].forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
            return None, first_hidden_states
        elif model_type in ["qwenimage", "flux", "sd35"]:
            if model_type == "qwenimage":
                first_encoder_hidden_states, first_hidden_states = transformer_blocks[0].forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
            elif model_type == "flux":
                first_encoder_hidden_states, first_hidden_states = transformer_blocks[0].forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
            elif model_type == "sd35":
                first_encoder_hidden_states, first_hidden_states = transformer_blocks[0].forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=attention_kwargs,
                )
                if controlnet_block_samples is not None and transformer_blocks[0].context_pre_only is False:
                    interval_control = len(transformer_blocks) // len(controlnet_block_samples)
                    hidden_states = hidden_states + controlnet_block_samples[0 // interval_control]
            return first_encoder_hidden_states, first_hidden_states
        else:
            raise ValueError(f"DiCache unsupported model type: {model_type}")

    def apply_cache(
        self,
        model_type: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        transformer_blocks: Optional[torch.nn.ModuleList] = None,
        single_transformer_blocks: Optional[torch.nn.ModuleList] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples = True,
        controlnet_blocks_repeat: bool = False,
        sparse_attn: Optional[SparseAttn] = None
    ):
        current_step = self.get_current_step()
        
        # Process first block
        first_encoder_hidden_states, first_hidden_states = self._process_first_block(
            model_type, hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
            temb, image_rotary_emb, attention_kwargs, transformer_blocks, controlnet_block_samples
        )
        
        modulated_inp = first_hidden_states.clone()
        cache_key = self.get_cache_key()

        # judge whether to use cache
        if current_step <= self.config.warmup_steps or self.previous_modulated_input_dict[cache_key] is None:
            should_calc = True
            self.accumulated_rel_l1_distance_dict[cache_key] = 0
        else:
            rel_distance = ((modulated_inp - self.previous_modulated_input_dict[cache_key]).abs().mean() / 
                          self.previous_modulated_input_dict[cache_key].abs().mean()).cpu().item()
            self.accumulated_rel_l1_distance_dict[cache_key] += rel_distance
            
            if self.accumulated_rel_l1_distance_dict[cache_key] < self.config.threshold:
                should_calc = False
                print(f"Skipping calculation at step {current_step} with accumulated relative L1 distance: {self.accumulated_rel_l1_distance_dict[cache_key]:.4f}")
            else:
                should_calc = True
                self.accumulated_rel_l1_distance_dict[cache_key] = 0
                
        self.previous_modulated_input_dict[cache_key] = modulated_inp   

        if not should_calc:
            hidden_states += self.previous_residual_dict[cache_key]
        else:
            ori_hidden_states = hidden_states.clone()
            hidden_states = first_hidden_states.clone()
            
            if first_encoder_hidden_states is not None:
                encoder_hidden_states = first_encoder_hidden_states.clone()
            
            remaining_blocks = transformer_blocks[1:] if len(transformer_blocks) > 1 else []
            
            if model_type == "wan":
                hidden_states = self._forward_wan_blocks(remaining_blocks, hidden_states, encoder_hidden_states, temb, image_rotary_emb, sparse_attn)
            elif model_type == "flux":
                encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                    model_type, remaining_blocks, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs,
                    controlnet_block_samples, controlnet_blocks_repeat, start_idx=1
                )
                
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                
                for index_block, block in enumerate(single_transformer_blocks):
                    hidden_states = block.forward(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                    if controlnet_single_block_samples is not None:
                        interval_control = len(single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )
                        
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
            elif model_type in ["qwenimage", "sd35"]:
                encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                    model_type, remaining_blocks, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs
                )
                
                if model_type == "sd35" and controlnet_block_samples is not None:
                    interval_control = len(transformer_blocks) // len(controlnet_block_samples)
                    hidden_states = hidden_states + controlnet_block_samples[-1 // interval_control]

            self.previous_residual_dict[cache_key] = hidden_states - ori_hidden_states

        return hidden_states

@AutoCache.register("dicache")
class DiCache(AutoCache):
    def __init__(self, config: DiCacheConfig):
        super().__init__(config)
        self.previous_probe_stats_dict = {
            "positive": None,
            "negative": None
        } # previous probe block output
        self.previous_probe_residual_dict = {
            "positive": None,
            "negative": None
        } # previous probe block residual
        self.previous_residual_window_dict = {
            "positive": [],
            "negative": []
        }
        self.previous_probe_residual_window_dict = {
            "positive": [],
            "negative": []
        }

    def apply_cache(
        self,
        model_type: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        transformer_blocks: Optional[torch.nn.ModuleList] = None,
        single_transformer_blocks: Optional[torch.nn.ModuleList] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples = True,
        controlnet_blocks_repeat: bool = False,
        sparse_attn: Optional[SparseAttn] = None
    ):
        current_step = self.get_current_step()
        total_steps = self.config.total_steps_callback() if self.config.total_steps_callback() is not None else 25

        cache_key = self.get_cache_key()


        ori_hidden_states = hidden_states.clone()
        probe_blocks = transformer_blocks[0:self.config.probe_depth]
        
        # Process probe blocks
        if model_type == "wan":
            hidden_states = self._forward_wan_blocks(probe_blocks, hidden_states, encoder_hidden_states, temb, image_rotary_emb)
        else:
            encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                model_type, probe_blocks, hidden_states, encoder_hidden_states,
                encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs
            )
        
        # judge whether to use cache
        if current_step <= int(self.config.ret_ratio * total_steps):
            should_calc = True
            self.accumulated_rel_l1_distance_dict[cache_key] = 0
        else:
            delta_x = (hidden_states - self.previous_modulated_input_dict[cache_key]).abs().mean() / self.previous_modulated_input_dict[cache_key].abs().mean() 
            delta_y = (hidden_states - self.previous_probe_stats_dict[cache_key]).abs().mean() / self.previous_probe_stats_dict[cache_key].abs().mean() 

            if self.config.rel_l1_distance_algo =="delta_minus":
                error = (delta_y - delta_x).abs()
            elif self.config.rel_l1_distance_algo == "delta_y":
                error = delta_y
            
            self.accumulated_rel_l1_distance_dict[cache_key] += error

            if self.accumulated_rel_l1_distance_dict[cache_key] < self.config.threshold: # skip this step
                should_calc = False 
                print(f"Skipping calculation at step {current_step} with accumulated relative L1 distance: {self.accumulated_rel_l1_distance_dict[cache_key]:.4f}")
            else:
                should_calc = True
                self.accumulated_rel_l1_distance_dict[cache_key] = 0

        self.previous_probe_stats_dict[cache_key] = hidden_states.clone()
        self.previous_modulated_input_dict[cache_key] = ori_hidden_states

        if not should_calc:
            if len(self.previous_residual_window_dict[cache_key]) >= 2:
                current_residual_indicator = torch.zeros_like(hidden_states)
                gamma = ((current_residual_indicator - self.previous_residual_window_dict[cache_key][-2]).abs().mean() / 
                        (self.previous_residual_window_dict[cache_key][-1] - self.previous_residual_window_dict[cache_key][-2]).abs().mean()).clip(1, 1.5)
                hidden_states += self.previous_residual_window_dict[cache_key][-2] + gamma * (
                    self.previous_residual_window_dict[cache_key][-1] - self.previous_residual_window_dict[cache_key][-2]
                )
            else:
                hidden_states += self.previous_residual_dict[cache_key] 
        else:
            unpass_transformer_blocks = transformer_blocks[self.config.probe_depth:]
            
            if model_type == "wan":
                hidden_states = self._forward_wan_blocks(unpass_transformer_blocks, hidden_states, encoder_hidden_states, temb, image_rotary_emb)
            elif model_type == "flux":
                encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                    model_type, unpass_transformer_blocks, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs,
                    controlnet_block_samples, controlnet_blocks_repeat, start_idx=self.config.probe_depth
                )
                
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                
                for index_block, block in enumerate(single_transformer_blocks):
                    hidden_states = block.forward(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                    if controlnet_single_block_samples is not None:
                        interval_control = len(single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1]:, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1]:, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )
                        
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, ...]
            else:
                encoder_hidden_states, hidden_states = self._forward_transformer_blocks(
                    model_type, unpass_transformer_blocks, hidden_states, encoder_hidden_states,
                    encoder_hidden_states_mask, temb, image_rotary_emb, attention_kwargs
                )
                
                if model_type == "sd35" and controlnet_block_samples is not None:
                    interval_control = len(transformer_blocks) // len(controlnet_block_samples)
                    hidden_states = hidden_states + controlnet_block_samples[-1 // interval_control]

            self.previous_residual_dict[cache_key] = hidden_states - ori_hidden_states
            self.previous_probe_residual_dict[cache_key] = self.previous_probe_stats_dict[cache_key] - ori_hidden_states

            self.previous_residual_window_dict[cache_key].append(self.previous_residual_dict[cache_key])
            self.previous_probe_residual_window_dict[cache_key].append(self.previous_probe_residual_dict[cache_key])
        
        return hidden_states
