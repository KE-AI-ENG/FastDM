import os
import json

from dataclasses import dataclass, fields
from typing import Optional, Type, Dict, Any

@dataclass
class SparseConfig:
    r"""Common configuration for sparse attention algorithms on diffusion models."""
    # common args
    sparse_algorithm: str
    enable_sparse: bool = False
    block_size: int = 128

    _registry: Dict[str, Type["SparseConfig"]] = None

    @classmethod
    def register(cls, name: str):
        def decorator(sub_cls):
            if cls._registry is None:
                cls._registry = {}
            cls._registry[name.lower()] = sub_cls
            return sub_cls
        return decorator

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseConfig":
        algo = data.get("sparse_algorithm")
        if algo is None:
            raise ValueError("sparse_algorithm must be specified in the config data.")
        algo = algo.lower()
        target_cls = cls._registry.get(algo, cls)

        # filter out keys that are not in the target class fields
        field_names = {f.name for f in fields(target_cls)}
        valid_keys = {k: v for k, v in data.items() if k in field_names}

        return target_cls(**valid_keys)

    @classmethod
    def from_json(cls, file_path: str) -> "SparseConfig":
        r"""Load configuration from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file is not exist: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

@SparseConfig.register("radial")
@dataclass
class RadialAttnConfig(SparseConfig):
    r"""Configuration for Radial Attention sparse algorithm."""
    backend: str = "sparse_sageattn"
    decay_factor: float = 0.5
    dense_layers: int = 1
    dense_steps: int = 5
    model_type: str = "wan"
    video_token_num: int = 25440
    num_frame: int = 16
    current_steps_callback: Optional[callable] = None





