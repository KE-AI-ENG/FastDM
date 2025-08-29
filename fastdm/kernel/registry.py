import os
import functools
from typing import Dict, Callable

class KernelRegistry:
    def __init__(self):
        self._registry: Dict[str, Dict[str, Callable]] = {}
    
    def register(self, op_name: str, backend: str):
        """Register the backend implementation of a specific operator"""
        def decorator(kernel_fn):
            if op_name not in self._registry:
                self._registry[op_name] = {}
            # Ensure that operators of the same backend are not registered repeatedly
            if backend in self._registry[op_name]:
                raise RuntimeError(f"Kernel {op_name} already registered for backend {backend}")
            self._registry[op_name][backend] = kernel_fn
        return decorator
    
    def select_backend(self, op_name: str, fallback="torch") -> str:
        """Select the backend implementation"""
        # 1. environment variable
        env_backend = os.environ.get('KERNEL_BACKEND', '').lower()
        if env_backend in self._registry.get(op_name, {}):
            return env_backend
        
        # 2. Heuristic rules (can be extended according to actual needs)
        # For example, automatically select the optimal backend based on GPU model, input size, etc.
        
        # 3. Fall back to the default implementation
        return fallback if fallback in self._registry.get(op_name, {}) else "torch"
    
    def dispatch(self, op_name: str, force_backend: str = None):
        """Create a distribution decorator"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Obtain the input tensor for heuristic selection
                # tensor_args = [a for a in args[:2] if isinstance(a, torch.Tensor)]
                # input_tensor = tensor_args[0] if tensor_args else None
                
                # if input_tensor is None:
                #     raise ValueError("Unable to recognize the input tensor for backend selection")
                
                backend = self.select_backend(op_name) if force_backend is None else force_backend
                backend_impl = self._registry[op_name].get(backend)
                
                if not backend_impl:
                    available = list(self._registry[op_name].keys())
                    raise RuntimeError(
                        f"kernel '{op_name}' backend '{backend}' unregistered. Available backends: {available}"
                    )
                
                return backend_impl(*args, **kwargs)
            return wrapper
        return decorator

# Global kernel regedit instance
kernel_registry = KernelRegistry()