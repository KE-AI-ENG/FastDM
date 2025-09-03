import os
import json

from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseCache:
    r"""Configuration for caching techniques on diffusion models."""
    enable_caching: bool = False
    threshold: float = 0.25
    current_steps_callback: Optional[callable] = None
    total_steps_callback: Optional[callable] = None  # total steps for the diffusion process
    coefficients: Optional[list] = None # coefficients teacache
    negtive_cache: Optional[bool] = False # qwenimage will excute the forward separately in one step for prompt and negative prompt. 
    negtive_coefficients: Optional[list] = None # coefficients for negative prompt teacache

    @classmethod
    def from_json(cls, file_path: str) -> "BaseCache":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"file is not exsit: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # filter out keys that are not in the class annotations
        valid_keys = {k: v for k, v in data.items() if k in cls.__annotations__}

        return cls(**valid_keys)