import torch
import torch.nn.functional as F
from typing import Optional, Type, Dict, Any
import numpy as np

from fastdm.caching.config import CacheConfig, TeaCacheConfig, FBCacheConfig, DiCacheConfig

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
    ):
        pass

@AutoCache.register("teacache")
class TeaCache(AutoCache):
    def __init__(self, config: TeaCacheConfig):
        super().__init__(config)
        self.coefficients = {
            "positive": config.coefficients,
            "negative": config.negtive_coefficients
        }

    def apply_cache(
        self,
        model_type: str,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        transformer_blocks: Optional[torch.nn.ModuleList] = None,
        single_transformer_blocks: Optional[torch.nn.ModuleList] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples = True,
        controlnet_blocks_repeat: bool = False,
    ):
        
        # get current step
        current_step = self.config.current_steps_callback() if self.config.current_steps_callback() is not None else 0

        # clone inputs
        inp = hidden_states.clone()
        inp1 = encoder_hidden_states.clone()
        temb_ = temb.clone()

        if model_type == "qwenimage":
            img_mod_params = transformer_blocks[0].img_mod_proj.forward(F.silu(temb_))
            txt_mod_params = transformer_blocks[0].txt_mod_proj.forward(F.silu(temb_))
            # Split modulation parameters for norm1 and norm2
            img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
            txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
            # Process image stream - norm1 + modulation
            img_normed = F.layer_norm(inp, (transformer_blocks[0].dim,), eps=transformer_blocks[0].eps)
            img_modulated, img_gate1 = transformer_blocks[0]._modulate(img_normed, img_mod1)
            # Process text stream - norm1 + modulation
            txt_normed = F.layer_norm(inp1, (transformer_blocks[0].dim,), eps=transformer_blocks[0].eps)
            txt_modulated, txt_gate1 = transformer_blocks[0]._modulate(txt_normed, txt_mod1)
            modulated_inp = txt_modulated.clone()
        elif model_type == "flux" or model_type == "sd35":
            modulated_inp, *_  = transformer_blocks[0].norm1.forward(inp, emb=temb_)
        else:
            raise ValueError(f"Teacache unsupported model type: {model_type}")

        # get cache key and update cache status
        if self.config.negtive_cache:
            cache_key = None
            for k in self.cache_status:
                if self.cache_status[k] and cache_key is None:
                    cache_key = k
                self.cache_status[k] = not self.cache_status[k]
        else:
            cache_key = "positive"

        # judge whether to use cache
        if current_step == 0:
            should_calc = True
            self.accumulated_rel_l1_distance_dict[cache_key] = 0
        else: 
            coefficients = self.coefficients[cache_key]
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance_dict[cache_key] += rescale_func(((modulated_inp-self.previous_modulated_input_dict[cache_key]).abs().mean() / self.previous_modulated_input_dict[cache_key].abs().mean()).cpu().item())
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
                for index_block, block in enumerate(transformer_blocks):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
            elif model_type == "flux":
                for index_block, block in enumerate(transformer_blocks):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                for index_block, block in enumerate(single_transformer_blocks):
                    hidden_states = block.forward(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )

                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            elif model_type == "sd35":
                for index_block, block in enumerate(transformer_blocks):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                # controlnet residual
                if controlnet_block_samples is not None and block.context_pre_only is False:
                    interval_control = len(transformer_blocks) // len(controlnet_block_samples)
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            self.previous_residual_dict[cache_key] = hidden_states - ori_hidden_states
            
        return hidden_states


@AutoCache.register("fbcache")
class FBCache(AutoCache):
    def __init__(self, config: FBCacheConfig):
        super().__init__(config)

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
    ):
        # get current step
        current_step = self.config.current_steps_callback() if self.config.current_steps_callback() is not None else 0

        # first block forward
        if model_type == "wan":
            first_hidden_states = transformer_blocks[0].forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
        elif model_type == "qwenimage":
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
            # controlnet residual
            if controlnet_block_samples is not None and transformer_blocks[0].context_pre_only is False:
                interval_control = len(transformer_blocks) // len(controlnet_block_samples)
                hidden_states = hidden_states + controlnet_block_samples[0 // interval_control]
        else:
            raise ValueError(f"FBCache unsupported model type: {model_type}")
        
        modulated_inp = first_hidden_states.clone()

        # get cache key and update cache status
        if self.config.negtive_cache:
            cache_key = None
            for k in self.cache_status:
                if self.cache_status[k] and cache_key is None:
                    cache_key = k
                self.cache_status[k] = not self.cache_status[k]
        else:
            cache_key = "positive"

        # judge whether to use cache
        if current_step == 0:
            should_calc = True
            self.accumulated_rel_l1_distance_dict[cache_key] = 0
        else: 
            self.accumulated_rel_l1_distance_dict[cache_key] += ((modulated_inp-self.previous_modulated_input_dict[cache_key]).abs().mean() / self.previous_modulated_input_dict[cache_key].abs().mean()).cpu().item()
            
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
            encoder_hidden_states = first_encoder_hidden_states.clone()
            if model_type == "wan":
                for block in transformer_blocks[1:]:
                    hidden_states = block.forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
            elif model_type == "qwenimage":
                for index_block, block in enumerate(transformer_blocks[1:]):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
            elif model_type == "flux":
                for index_block, block in enumerate(transformer_blocks[1:]):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                for index_block, block in enumerate(single_transformer_blocks):
                    hidden_states = block.forward(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )

                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            elif model_type == "sd35":
                for index_block, block in enumerate(self.transformer_blocks[1:]):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                # controlnet residual
                if controlnet_block_samples is not None and block.context_pre_only is False:
                    interval_control = len(self.transformer_blocks) // len(controlnet_block_samples)
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

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
    ):
        # get current step
        current_step = self.config.current_steps_callback() if self.config.current_steps_callback() is not None else 0
        total_steps = self.config.total_steps_callback() if self.config.total_steps_callback() is not None else 25

        # get cache key and update cache status
        if self.config.negtive_cache:
            cache_key = None
            for k in self.cache_status:
                if self.cache_status[k] and cache_key is None:
                    cache_key = k
                self.cache_status[k] = not self.cache_status[k]
        else:
            cache_key = "positive"


        ori_hidden_states, ori_encoder_hidden_states = hidden_states.clone(), encoder_hidden_states.clone()
        probe_blocks = transformer_blocks[0:self.config.probe_depth]
        # first block forward
        if model_type == "wan":
            for probe_block in probe_blocks:
                hidden_states = probe_block.forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
        elif model_type == "qwenimage":
            for probe_block in probe_blocks:
                encoder_hidden_states, hidden_states = probe_block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
        elif model_type == "flux":
            for probe_block in probe_blocks:
                encoder_hidden_states, hidden_states = probe_block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )
        elif model_type == "sd35":
            for probe_block in probe_blocks:
                encoder_hidden_states, hidden_states = probe_block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=attention_kwargs,
                )
        else:
            raise ValueError(f"FBCache unsupported model type: {model_type}")
        
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
                current_residual_indicator = hidden_states - hidden_states 
                gamma = ((current_residual_indicator - self.previous_residual_window_dict[cache_key][-2]).abs().mean() / (self.previous_residual_window_dict[cache_key][-1] - self.previous_residual_window_dict[cache_key][-2]).abs().mean()).clip(1, 1.5)
                hidden_states += self.previous_residual_window_dict[cache_key][-2] + gamma * (self.previous_residual_window_dict[cache_key][-1] - self.previous_residual_window_dict[cache_key][-2]) # 命中缓存时，使用历史的残差来预测当前的残差。
            else:
                hidden_states += self.previous_residual_dict[cache_key] 
        else:
            unpass_transformer_blocks = transformer_blocks[self.config.probe_depth:]
            if model_type == "wan":
                for block in unpass_transformer_blocks:
                    hidden_states = block.forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb)
            elif model_type == "qwenimage":
                for index_block, block in enumerate(unpass_transformer_blocks):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
            elif model_type == "flux":
                for index_block, block in enumerate(unpass_transformer_blocks):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                    # controlnet residual
                    if controlnet_block_samples is not None:
                        interval_control = len(transformer_blocks) / len(controlnet_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        # For Xlabs ControlNet.
                        if controlnet_blocks_repeat:
                            hidden_states = (
                                hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                            )
                        else:
                            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
                for index_block, block in enumerate(single_transformer_blocks):
                    hidden_states = block.forward(
                        hidden_states=hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=attention_kwargs,
                    )

                    # controlnet residual
                    if controlnet_single_block_samples is not None:
                        interval_control = len(single_transformer_blocks) / len(controlnet_single_block_samples)
                        interval_control = int(np.ceil(interval_control))
                        hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                            hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                            + controlnet_single_block_samples[index_block // interval_control]
                        )
                hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            elif model_type == "sd35":
                for index_block, block in enumerate(transformer_blocks[self.config.probe_depth:]):
                    encoder_hidden_states, hidden_states = block.forward(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=attention_kwargs,
                    )
                # controlnet residual
                if controlnet_block_samples is not None and block.context_pre_only is False:
                    interval_control = len(transformer_blocks) // len(controlnet_block_samples)
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

            self.previous_residual_dict[cache_key] = hidden_states - ori_hidden_states
            self.previous_probe_residual_dict[cache_key] = self.previous_probe_stats_dict[cache_key] - hidden_states

            self.previous_residual_window_dict[cache_key].append(self.previous_residual_dict[cache_key])
            self.previous_probe_residual_window_dict[cache_key].append(self.previous_probe_residual_dict[cache_key])
        
        return hidden_states
