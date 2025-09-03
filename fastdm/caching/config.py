import os
import json

from dataclasses import dataclass, field, fields
from typing import Optional, Type, Dict, Any, List

@dataclass
class CacheConfig:
    r"""Common configuration for caching algo on diffusion models."""
    # common args
    cache_algorithm : str
    enable_caching: bool = False
    threshold: float = 0.25
    current_steps_callback: Optional[callable] = None
    total_steps_callback: Optional[callable] = None
    negtive_cache: Optional[bool] = False # qwenimage/wan will excute the forward separately in one step for prompt and negative prompt. 

    _registry: Dict[str, Type["CacheConfig"]] = None

    @classmethod
    def register(cls, name: str):
        def decorator(sub_cls):
            if cls._registry is None:
                cls._registry = {}
            cls._registry[name.lower()] = sub_cls
            return sub_cls
        return decorator
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        algo = data.get("cache_algorithm")
        if algo is None:
            raise ValueError("cache_algorithm must be specified in the config data.")
        algo = algo.lower()
        target_cls = cls._registry.get(algo, cls)

        # filter out keys that are not in the target class fields
        field_names = {f.name for f in fields(target_cls)}
        valid_keys = {k: v for k, v in data.items() if k in field_names}

        return target_cls(**valid_keys)
    
    @classmethod
    def from_json(cls, file_path: str) -> "CacheConfig":
        r"""Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file is not exist: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)
    
@CacheConfig.register("teacache")
@dataclass
class TeaCacheConfig(CacheConfig):
    r"""Configuration for TeaCache algo on diffusion models."""
    coefficients: List[float] = field(default_factory=list) # coefficients teacache
    negtive_coefficients: List[float] = field(default_factory=list) # coefficients for negative prompt teacache

@CacheConfig.register("dicache")
@dataclass
class DiCacheConfig(CacheConfig):
    # dicache args
    probe_depth: int = 1
    ret_ratio: float = 0.2
    rel_l1_distance_algo: str = "delta_y" # delta_y, delta_minus

@CacheConfig.register("fbcache")
@dataclass
class FBCacheConfig(CacheConfig):
    pass

