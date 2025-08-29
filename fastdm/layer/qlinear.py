import torch

from fastdm.kernel.operators_set import quantize_to_fp8, quantize_to_int8, fp8_matmul, int8_matmul
from fastdm.utils.quantization import int8_quantization, fp8_quantization

class QLinear:
    def __init__(self, in_features, out_features, bias=True, data_type=torch.bfloat16, device_type="cuda"):
        super().__init__()
        self.weight = torch.empty((in_features, out_features), dtype=data_type)
        self.bias = torch.empty((out_features), dtype=data_type) if bias else None
        self.device = device_type
        self.has_bias = bias

        #for quantization
        self.weight_quant_scale = None
        self.weight_asym_sumcol = None  # for int8 asymmetric quantization

    def weight_loading_and_quant(self, src_weight, src_bias, quant_type = None):
        """
        Load the weight and bias from the source tensors.
        """
        if len(src_weight) > 1:#mm fusion
            '''
            we need a col-major matrix to call matrix-mul, so we need to do `.contiguous().transpose(0,1).contiguous().transpose(0,1)`.
            the first contiguous to create an new row-major tensor, its shape is kxn, 
            the transpose(0,1).contiguous() is to create an new nxk tensor, it is row-major(the k dim is contiguous)
            the last transpose is to creat a col-major tensor, its shape is kxn which can correctly do (mxk @ kxn)-matmul.
            '''
            self.weight = torch.cat(src_weight, 1).contiguous().transpose(0,1).contiguous().transpose(0,1)
            if self.has_bias:
                self.bias = torch.cat(src_bias, 0).contiguous().to(self.device)
        else:
            self.weight = src_weight[0].to(self.weight.dtype)
            if src_bias[0] is not None:
                assert self.has_bias, "Bias is not enabled but bias tensor is provided."
                self.bias = src_bias[0].to(self.bias.dtype).to(self.device)
        
        # quantization
        if quant_type is not None:
            if quant_type == torch.float8_e4m3fn:
                #weights shape is (k,n), we need to transpose it to (n,k) to get a (n,) scales and a (k,n) quantized tensor
                self.weight, self.weight_quant_scale = fp8_quantization(self.weight.transpose(0,1).contiguous())
                self.weight = self.weight.transpose(0,1).to(self.device)
                self.weight_quant_scale = self.weight_quant_scale.to(self.device)
            elif quant_type == torch.int8:
                #weights shape is (k,n), we need to transpose it to (n,k) to get a (n,) scales and a (k,n) quantized tensor
                self.weight, self.weight_quant_scale, _ = int8_quantization(self.weight.transpose(0,1).contiguous())
                self.weight = self.weight.transpose(0,1).to(self.device)
                self.weight_asym_sumcol = self.weight.to(torch.int32).sum(dim=0, keepdim=True, dtype=torch.int32) # for asymmetric quantization matrix multiplication
                self.weight_quant_scale = self.weight_quant_scale.to(self.device)
            else:
                raise ValueError(f"Unsupported quantization type: {quant_type}")
        else:
            self.weight = self.weight.to(self.device)

    def forward(self, input_tensor):
        """
        Perform the forward pass of the qlinear layer with quantization support.
        Args:
            input_tensor (torch.Tensor): The input tensor(bf16/fp16) to the linear layer.
        
        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.
        """
        ori_shape = input_tensor.shape
        if len(ori_shape)>2:
            input_tensor = input_tensor.reshape(-1, ori_shape[-1])

        if torch.float8_e4m3fn == self.weight.dtype:
            x_quant, x_scale = quantize_to_fp8(input_tensor)
            output_tensor = fp8_matmul(x_quant, self.weight, x_scale, self.weight_quant_scale, input_tensor.dtype, bias=self.bias)
        elif torch.int8 == self.weight.dtype:
            x_quant, x_scale, x_zp = quantize_to_int8(input_tensor, symmetric=False)
            output_tensor = int8_matmul(x_quant, self.weight, x_scale, self.weight_quant_scale, input_tensor.dtype, self.weight_asym_sumcol, x_zp, bias=self.bias)
        else:
            output_tensor = torch.addmm(self.bias, input_tensor, self.weight) if self.bias is not None else torch.mm(input_tensor, self.weight)
        
        if len(ori_shape)>2:
            output_tensor = output_tensor.view(*(ori_shape[:-1]), self.weight.shape[-1])

        return output_tensor

