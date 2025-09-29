import torch
# import flashinfer
from einops import repeat
from typing import Type, Dict, Any
from fastdm.sparse.config import SparseConfig, RadialAttnConfig
from fastdm.kernel.operators_set import scaled_dot_product_attention, sparse_scaled_dot_product_attention
class SparseAttn:
    _registry: Dict[str, Type["SparseAttn"]] = {}

    def __init__(self, config: SparseConfig):
        self.config = config

    @classmethod
    def register(cls, name: str):
        def decorator(sub_cls):
            cls._registry[name.lower()] = sub_cls
            return sub_cls
        return decorator

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseAttn":
        """Create a sparse instance from a dictionary."""
        config = SparseConfig.from_dict(data)
        algo = config.sparse_algorithm.lower()
        sparse_cls = cls._registry.get(algo)
        if sparse_cls is None:
            raise ValueError(f"Unknown sparse algorithm: {algo}")
        return sparse_cls(config)

    @classmethod
    def from_json(cls, file_path: str) -> "SparseAttn":
        """Load sparse configuration from a JSON file and create a sparse instance."""
        config = SparseConfig.from_json(file_path)
        algo = config.sparse_algorithm.lower()
        sparse_cls = cls._registry.get(algo)
        if sparse_cls is None:
            raise ValueError(f"Unknown sparse algorithm: {algo}")
        return sparse_cls(config)

    def apply(self, query, key, value, pre_defined_mask=None):
        raise NotImplementedError("Subclasses must implement apply method")


@SparseAttn.register("radial")
class RadialAttn(SparseAttn):
    _log_mask = None

    def __init__(self, config: RadialAttnConfig):
        """
        Radial attention with a decay factor and dense layers.
        Args:
            config (RadialAttnConfig): Configuration object for radial sparse attention.
        """
        super().__init__(config)

    def post_init(self, video_token_num=None, num_frame=None):
        self.video_token_num = video_token_num
        self.num_frame = num_frame

    def apply(self, query, key, value, pre_defined_mask=None):
        video_mask = self.queryLogMask(query)
        return self.SpargeSageAttnBackend(query, key, value, video_mask, pre_defined_mask)
            
    def queryLogMask(self, query):
        if RadialAttn._log_mask is None:
            RadialAttn._log_mask = torch.ones((query.shape[0] // self.config.block_size, query.shape[0] // self.config.block_size), device=query.device, dtype=torch.bool)
            RadialAttn._log_mask = self.gen_log_mask_shrinked(query.shape[0], query.device)
        return RadialAttn._log_mask


    def gen_log_mask_shrinked(self, s, device):
        """
        A more memory friendly version, we generate the attention mask of each frame pair at a time,
        shrinks it, and stores it into the final result
        """
        final_log_mask = torch.zeros((s // self.config.block_size, s // self.config.block_size), device=device, dtype=torch.bool)
        token_per_frame = self.video_token_num // self.num_frame
        video_text_border = self.video_token_num // self.config.block_size

        col_indices = torch.arange(0, token_per_frame, device=device).view(1, -1)
        row_indices = torch.arange(0, token_per_frame, device=device).view(-1, 1)
        final_log_mask[video_text_border:] = True
        final_log_mask[:, video_text_border:] = True
        for i in range(self.num_frame):
            for j in range(self.num_frame):
                local_mask = torch.zeros((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
                if j == 0 and self.config.model_type == "wan": # this is attention sink
                    local_mask = torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
                else:
                    window_width = self.get_window_width(i, j, token_per_frame)
                    local_mask = torch.abs(col_indices - row_indices) <= window_width
                    split_mask = self.get_diagonal_split_mask(i, j, token_per_frame, device)
                    local_mask = torch.logical_and(local_mask, split_mask)

                remainder_row = (i * token_per_frame) % self.config.block_size
                remainder_col = (j * token_per_frame) % self.config.block_size
                # get the padded size
                all_length_row = remainder_row + ((token_per_frame - 1) // self.config.block_size + 1) * self.config.block_size
                all_length_col = remainder_col + ((token_per_frame - 1) // self.config.block_size + 1) * self.config.block_size
                padded_local_mask = torch.zeros((all_length_row, all_length_col), device=device, dtype=torch.bool)
                padded_local_mask[remainder_row:remainder_row + token_per_frame, remainder_col:remainder_col + token_per_frame] = local_mask
                # shrink the mask
                block_mask = self.shrinkMaskStrict(padded_local_mask)
                # set the block mask to the final log mask
                block_row_start = (i * token_per_frame) // self.config.block_size
                block_col_start = (j * token_per_frame) // self.config.block_size
                block_row_end = block_row_start + block_mask.shape[0]
                block_col_end = block_col_start + block_mask.shape[1]
                final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                    final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask)
        print(f"mask sparsity: {1 - final_log_mask.sum() / final_log_mask.numel()}")
        return final_log_mask


    def get_diagonal_split_mask(self, i, j, token_per_frame, device):
        dist = abs(i - j)
        group = dist.bit_length()
        threshold = self.config.block_size
        decay_length = 2 ** token_per_frame.bit_length() / 2 ** group
        if decay_length >= threshold:
            return torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)

        split_factor = int(threshold / decay_length)
        modular = dist % split_factor
        if modular == 0:
            return torch.ones((token_per_frame, token_per_frame), device=device, dtype=torch.bool)
        else:
            return torch.zeros((token_per_frame, token_per_frame), device=device, dtype=torch.bool)

    def get_window_width(self, i, j, token_per_frame):
        dist = abs(i - j)
        if self.config.model_type == "wan":
            if dist < 1:
                return token_per_frame
            if dist == 1:
                return token_per_frame // 2
        elif self.config.model_type == "hunyuan":
            if dist <= 1:
                return token_per_frame
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        group = dist.bit_length()
        decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * self.config.decay_factor
        threshold = self.config.block_size
        if decay_length >= threshold:
            return decay_length
        else:
            return threshold
        
    def get_indptr_from_mask(self, mask, query):
        # query shows the device of the indptr
        # indptr (torch.Tensor) - the block index pointer of the block-sparse matrix on row dimension,
        # shape `(MB + 1,)`, where `MB` is the number of blocks in the row dimension.
        # The first element is always 0, and the last element is the number of blocks in the row dimension.
        # The rest of the elements are the number of blocks in each row.
        # the mask is already a block sparse mask
        indptr = torch.zeros(mask.shape[0] + 1, device=query.device, dtype=torch.int32)
        indptr[0] = 0
        row_counts = mask.sum(dim=1).flatten()  # Ensure 1D output [num_blocks_row]
        indptr[1:] = torch.cumsum(row_counts, dim=0)
        return indptr

    def get_indices_from_mask(self, mask, query):
        # indices (torch.Tensor) - the block indices of the block-sparse matrix on column dimension,
        # shape `(nnz,),` where `nnz` is the number of non-zero blocks.
        # The elements in `indices` array should be less than `NB`: the number of blocks in the column dimension.
        nonzero_indices = torch.nonzero(mask)
        indices = nonzero_indices[:, 1].to(dtype=torch.int32, device=query.device)
        return indices

    def shrinkMaskStrict(self, mask):
        seqlen = mask.shape[0]
        block_num = seqlen // self.config.block_size
        mask = mask[:block_num * self.config.block_size, :block_num * self.config.block_size].view(block_num, self.config.block_size, block_num, self.config.block_size)
        col_densities = mask.sum(dim = 1) / self.config.block_size
        # we want the minimum non-zero column density in the block
        non_zero_densities = col_densities > 0
        high_density_cols = col_densities > 1/3
        frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
        block_mask = frac_high_density_cols > 0.6
        block_mask[0:0] = True
        block_mask[-1:-1] = True
        return block_mask
    
    def SpargeSageAttnBackend(self, query, key, value, video_mask=None, pre_defined_mask=None):
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        b, t, h, d = query.shape
        arch = get_cuda_arch_versions()[query.device.index]
        converted_mask = repeat(sparge_mask_convert(mask=video_mask, block_size=self.config.block_size, arch=arch), "s t -> b h s t", b=b, h=h)
        
        converted_mask = converted_mask.to(torch.int8)
        if pre_defined_mask is None:
            # wan case
            output = sparse_scaled_dot_product_attention(
                query[:, :self.video_token_num, :, :].reshape(b, -1, h*d),
                key[:, :self.video_token_num, :, :].reshape(b, -1, h*d),
                value[:, :self.video_token_num, :, :].reshape(b, -1, h*d),
                num_q_heads=h, num_kv_heads=h, head_dim=d, is_causal=False, scale=1.0/(query.shape[-1]**0.5), sparse_mask=converted_mask)

            output = output.reshape(t, h, d)    # b=1
            return output
        
        kv_border = (pre_defined_mask[0].sum() + 63) // 64
        converted_mask[:, :, :, kv_border:] = False
        output_video = sparse_scaled_dot_product_attention(
                query[:, :self.video_token_num, :, :].reshape(b, -1, h*d), 
                key.reshape(b, -1, h*d), 
                value.reshape(b, -1, h*d),
                num_q_heads=h, num_kv_heads=h, head_dim=d, is_causal=False, scale=1.0/(query.shape[-1]**0.5), sparse_mask=converted_mask[:, :, :self.video_token_num // self.config.block_size, :].contiguous())
        
        output_video = output_video.reshape(self.video_token_num, h, d)    # b=1
        
        output_text = scaled_dot_product_attention(
                query[:, self.video_token_num:, :, :].reshape(b, -1, h*d),
                key[:, :pre_defined_mask[0].sum(), :, :].reshape(b, -1, h*d),
                value[:, :pre_defined_mask[0].sum(), :, :].reshape(b, -1, h*d),
                num_q_heads=h, num_kv_heads=h, head_dim=d, is_causal=False, scale=1.0/(query.shape[-1]**0.5))
        output_text = output_text.reshape(t-self.video_token_num, h, d) # b=1
        return torch.cat([output_video, output_text], dim=0)
        
def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs

def sparge_mask_convert(mask: torch.Tensor, block_size: int = 128, arch="sm") -> torch.Tensor:
    assert block_size in [128, 64], "Radial Attention only supports block size of 128 or 64"
    assert mask.shape[0] == mask.shape[1], "Input mask must be square."

    if block_size == 128:
        if arch == "sm90":
            new_mask = torch.repeat_interleave(mask, 2, dim=0)
        else:
            new_mask = torch.repeat_interleave(mask, 2, dim=1)
        
    elif block_size == 64:
        if arch == "sm90":
            num_row, num_col = mask.shape
            reshaped_mask = mask.view(num_row, num_col // 2, 2)
            new_mask = torch.max(reshaped_mask, dim=2).values
        else:
            num_row, num_col = mask.shape
            reshaped_mask = mask.view(num_row // 2, 2, num_col)
            new_mask = torch.max(reshaped_mask, dim=1).values

    return new_mask
