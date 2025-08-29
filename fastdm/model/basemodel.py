import torch

from fastdm.layer.qlinear import QLinear

class BaseModelCore:
    def __init__(self, type: str="DiT"):
        self.model_type = type

    def init_weight(self, weights_name, dst_module=None, quant_dtype=None):
        '''
        Initialize weights from origin_tensor_dict to dst_module.

        Args:
            weights_name: list of weight names to be initialized
            dst_module: destination module to load weights. if None, loading norm or conv weights, return an new tensor. if Linear, load the weights and quantize it.
            quant_dtype: quantization data type, if None, no quantization will be applied
        Returns:
            dst_tensor: the tensor loaded to dst_module, if dst_module is Linear, do weight loading in the module's function, return None
        '''
        if None == dst_module: #norm weight or conv weight
            assert 1 == len(weights_name)
            weight_name = weights_name[0]
            if weight_name not in self.origin_tensor_dict.keys():
                raise ValueError(f"The {weight_name} is not in origin_tensor_dict!!!")
            src_tensor = self.origin_tensor_dict[weight_name]
            dst_tensor = src_tensor.to(self.device)
            del src_tensor
            self.unmatched_tensors.remove(weight_name)
            return dst_tensor
        elif isinstance(dst_module, QLinear): #Linear weight
            src_w_list = []
            src_b_list = []
            src_w_name_list = []
            src_b_name_list = []
            for weight_name in weights_name:
                w_name = f"{weight_name}.weight"
                b_name = f"{weight_name}.bias"
                if w_name not in self.origin_tensor_dict.keys():
                    raise ValueError(f"The {w_name} is not in origin_tensor_dict!!!")
                src_w = self.origin_tensor_dict[w_name]
                src_b = self.origin_tensor_dict[b_name] if b_name in self.origin_tensor_dict.keys() else None
                src_w_list.append(src_w.transpose(0,1)) #linear weights need (in_features, out_features) shape
                src_b_list.append(src_b)
                src_w_name_list.append(w_name)
                src_b_name_list.append(b_name)
            
            dst_module.weight_loading_and_quant(src_w_list, src_b_list, quant_type=quant_dtype)

            for src_w, src_b, src_w_name, src_b_name in zip(src_w_list, src_b_list, src_w_name_list, src_b_name_list):
                del src_w
                self.unmatched_tensors.remove(src_w_name)
                if src_b is not None:
                    del src_b
                    self.unmatched_tensors.remove(src_b_name)

            return None

        else:
            raise ValueError(f"Unsupported dst_module type: {type(dst_module)} weights loading!")

    def _pre_part_loading(self):
        """
        Load pre-part weights, such as embedding and positional encoding.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    def _major_parts_loading(self):
        """
        Load major parts weights, such as transformer layers.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")
    
    def _post_part_loading(self):
        """
        Load post-part weights, such as output layers.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def weight_loading(self, model_path, data_type=torch.bfloat16, device_type="cuda"):
        
        """
        Load model weights from a given path into the model.

        Args:
            model_path: path to the model weights, can be a directory or a state_dict
            data_type: data type of the model weights, default is torch.bfloat16
            device_type: device type to load the model weights, default is "cuda"
        """

        self.origin_tensor_dict = {}
        self.dtype = data_type
        self.device = device_type

        if isinstance(model_path, dict):
            self.origin_tensor_dict = model_path
        else:
            import os
            if os.path.isdir(model_path): #dir
                for root, dirs, files in os.walk(model_path):
                    for name in files:
                        #read model_path
                        if ".safetensors" in name: #hf safetensors
                            file_path = os.path.join(root, name)
                            from safetensors import safe_open
                            with safe_open(file_path, framework="pt") as f:
                                for k in f.keys():
                                    self.origin_tensor_dict[k] = f.get_tensor(k)
            else: #file
                #read model_path
                if ".safetensors" in model_path: #hf safetensors
                    from safetensors import safe_open
                    with safe_open(model_path, framework="pt") as f:
                        for k in f.keys():
                            self.origin_tensor_dict[k] = f.get_tensor(k)
                else: #torch bin
                    self.origin_tensor_dict = torch.load(model_path, weights_only=True)

        self.unmatched_tensors = list(self.origin_tensor_dict.keys()) #Unmatched tensors ensure that all weight loads can be completed correctly

        self._pre_part_loading()
        self._major_parts_loading()
        self._post_part_loading()

        if not isinstance(model_path, dict):
            del self.origin_tensor_dict

        torch.cuda.empty_cache()

        assert len(self.unmatched_tensors) == 0

        return