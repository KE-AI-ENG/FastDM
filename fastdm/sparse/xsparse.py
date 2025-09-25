from typing import Optional
import torch
import flashinfer
from einops import rearrange, repeat
from sageattention import sageattn
try:
    from spas_sage_attn import block_sparse_sage2_attn_cuda
except ImportError:
    print("Using sparse_sageattn as block_sparse_sage2_attn_cuda")
    from sparse_sageattn import sparse_sageattn as block_sparse_sage2_attn_cuda

class SparseAttn:
    def __init__(self, block_size=128, model_type=None):
        self.block_size = block_size
        self.model_type = model_type
        
class RadialAttn(SparseAttn):    
    _log_mask = None

    def __init__(
            self, 
            block_size=128, 
            model_type=None, 
            decay_factor=0.5,
            dense_layers=1,
            dense_steps=5,
            backend="sparse_sageattn"):
        """
        Radial attention with a decay factor and dense layers.
        Args:
            block_size (int): The size of the blocks for block-sparse attention.
            model_type (str): The type of the model, e.g., "wan", "hunyuan".
            decay_factor (float): The factor by which the attention window decays.
            dense_layers (int): The number of dense layers to apply.
            dense_timesteps (int): The number of steps for dense layers.
        """
        super().__init__(block_size, model_type)
        self.decay_factor = decay_factor
        self.sparse_type = "radial"
        self.dense_layers = dense_layers
        self.dense_steps = dense_steps
        self.backend = backend
        self.current_steps_callback = None
    
    def post_init(self, video_token_num=25440, num_frame=16):
        self.video_token_num = video_token_num
        self.num_frame = num_frame

    def apply(self, query, key, value, pre_defined_mask=None):
        orig_seqlen, num_head, hidden_dim = query.shape
        video_mask = self.queryLogMask(query)
        
        if self.backend == "flashinfer":
            video_mask = video_mask[:self.video_token_num // self.block_size, :self.video_token_num // self.block_size]
            # perform block-sparse attention on the video tokens
            workspace_buffer = torch.empty(128 * 1024 * 1024, device=query.device, dtype=torch.uint8)
            bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(
                workspace_buffer,
                backend="fa3",
            )
            
            indptr = self.get_indptr_from_mask(video_mask, query)
            indices = self.get_indices_from_mask(video_mask, query)
            
            bsr_wrapper.plan(
                indptr=indptr,
                indices=indices,
                M=self.video_token_num,
                N=self.video_token_num,
                R=self.block_size,
                C=self.block_size,
                num_qo_heads=num_head,
                num_kv_heads=num_head,
                head_dim=hidden_dim,
                q_data_type=query.dtype,
                kv_data_type=key.dtype,
                o_data_type=query.dtype,
            )
            
            return self.FlashInferBackend(query, key, value,pre_defined_mask, bsr_wrapper)
        elif self.backend == "sparse_sageattn":
            return self.SpargeSageAttnBackend(query, key, value, video_mask, pre_defined_mask)
            
    def queryLogMask(self, query):
        if RadialAttn._log_mask is None:
            RadialAttn._log_mask = torch.ones((query.shape[0] // self.block_size, query.shape[0] // self.block_size), device=query.device, dtype=torch.bool)
            RadialAttn._log_mask = self.gen_log_mask_shrinked(query, query.shape[0], self.video_token_num, self.num_frame, sparse_type=self.sparse_type, decay_factor=self.decay_factor, model_type=self.model_type, block_size=self.block_size)
        return RadialAttn._log_mask


    def gen_log_mask_shrinked(self, query, s, video_token_num, num_frame, block_size=128, sparse_type="log", decay_factor=0.5, model_type=None):
        """
        A more memory friendly version, we generate the attention mask of each frame pair at a time,
        shrinks it, and stores it into the final result
        """
        final_log_mask = torch.zeros((s // block_size, s // block_size), device=query.device, dtype=torch.bool)
        token_per_frame = video_token_num // num_frame
        video_text_border = video_token_num // block_size

        col_indices = torch.arange(0, token_per_frame, device=query.device).view(1, -1)
        row_indices = torch.arange(0, token_per_frame, device=query.device).view(-1, 1)
        final_log_mask[video_text_border:] = True
        final_log_mask[:, video_text_border:] = True
        for i in range(num_frame):
            for j in range(num_frame):
                local_mask = torch.zeros((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
                if j == 0 and model_type == "wan": # this is attention sink
                    local_mask = torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
                else:
                    window_width = self.get_window_width(i, j, token_per_frame, sparse_type, num_frame, decay_factor=decay_factor, block_size=block_size, model_type=model_type)
                    local_mask = torch.abs(col_indices - row_indices) <= window_width
                    split_mask = self.get_diagonal_split_mask(i, j, token_per_frame, sparse_type, query)
                    local_mask = torch.logical_and(local_mask, split_mask)

                remainder_row = (i * token_per_frame) % block_size
                remainder_col = (j * token_per_frame) % block_size
                # get the padded size
                all_length_row = remainder_row + ((token_per_frame - 1) // block_size + 1) * block_size
                all_length_col = remainder_col + ((token_per_frame - 1) // block_size + 1) * block_size
                padded_local_mask = torch.zeros((all_length_row, all_length_col), device=query.device, dtype=torch.bool)
                padded_local_mask[remainder_row:remainder_row + token_per_frame, remainder_col:remainder_col + token_per_frame] = local_mask
                # shrink the mask
                block_mask = self.shrinkMaskStrict(padded_local_mask, block_size=block_size)
                # set the block mask to the final log mask
                block_row_start = (i * token_per_frame) // block_size
                block_col_start = (j * token_per_frame) // block_size
                block_row_end = block_row_start + block_mask.shape[0]
                block_col_end = block_col_start + block_mask.shape[1]
                final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end] = torch.logical_or(
                    final_log_mask[block_row_start:block_row_end, block_col_start:block_col_end], block_mask)
        print(f"mask sparsity: {1 - final_log_mask.sum() / final_log_mask.numel()}")
        return final_log_mask


    def get_diagonal_split_mask(self, i, j, token_per_frame, sparse_type, query):
        assert(sparse_type in ["radial"])
        dist = abs(i - j)
        group = dist.bit_length()
        threshold = 128 # hardcoded threshold for now, which is equal to block-size
        decay_length = 2 ** token_per_frame.bit_length() / 2 ** group
        if decay_length >= threshold:
            return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
        
        split_factor = int(threshold / decay_length)
        modular = dist % split_factor
        if modular == 0:
            return torch.ones((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)
        else:
            return torch.zeros((token_per_frame, token_per_frame), device=query.device, dtype=torch.bool)

    def get_window_width(self, i, j, token_per_frame, sparse_type, num_frame, decay_factor=1, block_size=128, model_type=None):
        assert(sparse_type in ["radial"])
        dist = abs(i - j)
        if model_type == "wan":
            if dist < 1:
                return token_per_frame
            if dist == 1:
                return token_per_frame // 2
        elif model_type == "hunyuan":
            if dist <= 1:
                return token_per_frame
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        group = dist.bit_length()
        decay_length = 2 ** token_per_frame.bit_length() / 2 ** group * decay_factor
        threshold = block_size
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

    def shrinkMaskStrict(self, mask, block_size=128):
        seqlen = mask.shape[0]
        block_num = seqlen // block_size
        mask = mask[:block_num * block_size, :block_num * block_size].view(block_num, block_size, block_num, block_size)
        col_densities = mask.sum(dim = 1) / block_size
        # we want the minimum non-zero column density in the block
        non_zero_densities = col_densities > 0
        high_density_cols = col_densities > 1/3
        frac_high_density_cols = high_density_cols.sum(dim=-1) / (non_zero_densities.sum(dim=-1) + 1e-9)
        block_mask = frac_high_density_cols > 0.6
        block_mask[0:0] = True
        block_mask[-1:-1] = True
        return block_mask
    
    def SpargeSageAttnBackend(self, query, key, value, video_mask=None, pre_defined_mask=None):
        if video_mask.all():
            # dense case
            kv_border = pre_defined_mask[0].sum() if pre_defined_mask is not None else key.shape[0]
            output_video = sageattn(
                query[:self.video_token_num, :, :].unsqueeze(0),
                key[:kv_border, :, :].unsqueeze(0),
                value[:kv_border, :, :].unsqueeze(0),
                tensor_layout="NHD",
            )[0]
            
            if pre_defined_mask is not None:
                output_text = flashinfer.single_prefill_with_kv_cache(
                    q=query[self.video_token_num:, :, :],
                    k=key[:pre_defined_mask[0].sum(), :, :],
                    v=value[:pre_defined_mask[0].sum(), :, :],
                    causal=False,
                    return_lse=False,
                )
                return torch.cat([output_video, output_text], dim=0)
            else:
                return output_video
        
        # sparse-sageattention only supports (b, h, s, d) layout, need rearrange first
        query_hnd = rearrange(query.unsqueeze(0), "b s h d -> b h s d")
        key_hnd = rearrange(key.unsqueeze(0), "b s h d -> b h s d")
        value_hnd = rearrange(value.unsqueeze(0), "b s h d -> b h s d")
        arch = get_cuda_arch_versions()[query.device.index]
        converted_mask = repeat(sparge_mask_convert(mask=video_mask, block_size=self.block_size, arch=arch), "s t -> b h s t", b=query_hnd.shape[0], h=query_hnd.shape[1])
        
        converted_mask = converted_mask.to(torch.int8)
        if pre_defined_mask is None:
            # wan case
            output = block_sparse_sage2_attn_cuda(
                query_hnd[:, :, :self.video_token_num, :],
                key_hnd[:, :, :self.video_token_num, :],
                value_hnd[:, :, :self.video_token_num, :],
                mask_id=converted_mask,
                tensor_layout="HND",
            )

            # rearrange back to (s, h, d), we know that b = 1
            output = rearrange(output, "b h s d -> s (b h) d", b=1)
            return output
        
        query_video = query_hnd[:, :, :self.video_token_num, :]
        key_video = key_hnd
        value_video = value_hnd
        kv_border = (pre_defined_mask[0].sum() + 63) // 64
        converted_mask[:, :, :, kv_border:] = False
        output_video = block_sparse_sage2_attn_cuda(
            query_video,
            key_video,
            value_video,
            mask_id=converted_mask[:, :, :self.video_token_num // self.block_size, :].contiguous(),
            tensor_layout="HND",
        )
        
        # rearrange back to (s, h, d), we know that b = 1
        output_video = rearrange(output_video, "b h s d -> s (b h) d", b=1)
        
        # gt = sparse_sageattn(
        #     query_video,
        #     key_video,
        #     value_video,
        #     mask_id=None,
        #     is_causal=False,
        #     tensor_layout="HND",
        # )[0]
        
        
        
        # import pdb; pdb.set_trace()
        
        output_text = flashinfer.single_prefill_with_kv_cache(
            q=query[self.video_token_num:, :, :],
            k=key[:pre_defined_mask[0].sum(), :, :],
            v=value[:pre_defined_mask[0].sum(), :, :],
            causal=False,
            return_lse=False,
        )
        
        return torch.cat([output_video, output_text], dim=0)
        

    def FlashInferBackend(self, query, key, value, pre_defined_mask=None, bsr_wrapper=None):
        if pre_defined_mask is not None:
            video_video_o, video_video_o_lse = bsr_wrapper.run(
                query[:self.video_token_num, :, :], 
                key[:self.video_token_num, :, :],
                value[:self.video_token_num, :, :],
                return_lse=True
            ) 
            # perform non-causal flashinfer on the text tokens
            video_text_o, video_text_o_lse = flashinfer.single_prefill_with_kv_cache(
                q=query[:self.video_token_num, :, :],
                k=key[self.video_token_num:, :, :],
                v=value[self.video_token_num:, :, :],
                causal=False,
                return_lse=True,
                custom_mask=pre_defined_mask[:self.video_token_num, self.video_token_num:]
            )
            
            # merge the two results
            o_video, _ = flashinfer.merge_state(v_a=video_video_o, s_a=video_video_o_lse, v_b=video_text_o, s_b=video_text_o_lse)
            
            o_text = flashinfer.single_prefill_with_kv_cache(
                q=query[self.video_token_num:, :, :],
                k=key,
                v=value,
                causal=False,
                return_lse=False,
                custom_mask=pre_defined_mask[self.video_token_num:, :]
            )
            
            return torch.cat([o_video, o_text], dim=0)
        else:
            o = bsr_wrapper.run(
                query[:self.video_token_num, :, :],
                key[:self.video_token_num, :, :],
                value[:self.video_token_num, :, :]
            )
            return o
        

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
