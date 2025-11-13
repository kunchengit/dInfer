import torch
from typing import Optional, Sequence

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)


class SglangKVCacheManager:
    """KV manager that allocates KV slots per block using SGLang pools."""

    def __init__(self, model_runner, max_length: int):
        self.model_runner = model_runner
        self.device = torch.device(model_runner.device)
        self.max_length = max_length
        self.initialized = False
        self.req_pool_indices: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.seq_lens_cpu: Optional[torch.Tensor] = None
        self.batch_size: Optional[int] = None
        self.target_length: Optional[int] = None

    def initialize(self, batch_size: int, total_length: int):
        if self.initialized:
            self.release()

        if total_length > self.max_length:
            raise ValueError(
                f"Requested total length {total_length} exceeds configured "
                f"maximum {self.max_length}"
            )

        req_indices = self.model_runner.req_to_token_pool.alloc(batch_size)
        if req_indices is None:
            raise RuntimeError("Not enough request slots for KV cache")

        self.req_pool_indices = torch.tensor(
            req_indices, dtype=torch.int32, device=self.device
        )
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        self.seq_lens_cpu = torch.zeros(batch_size, dtype=torch.int32, device="cpu")
        self.batch_size = batch_size
        self.target_length = total_length
        self.initialized = True

    def release(self):
        if not self.initialized:
            return

        req_tensor = self.model_runner.req_to_token_pool.req_to_token
        for row in range(self.batch_size):
            length = int(self.seq_lens[row].item())
            if length == 0:
                continue
            slots = (
                req_tensor[int(self.req_pool_indices[row].item()), :length]
                .to(torch.int64)
                .to(self.device)
            )
            self.model_runner.token_to_kv_pool_allocator.free(slots)

        self.model_runner.req_to_token_pool.free(
            self.req_pool_indices.detach().cpu().tolist()
        )

        self.initialized = False
        self.req_pool_indices = None
        self.seq_lens = None
        self.seq_lens_cpu = None
        self.batch_size = None
        self.target_length = None

    def _ensure_length(self, target_lens: torch.Tensor):
        needed = (target_lens - self.seq_lens).clamp_min_(0)
        if torch.all(needed == 0):
            return

        req_tensor = self.model_runner.req_to_token_pool.req_to_token
        for row in range(self.batch_size):
            add_len = int(needed[row].item())
            if add_len == 0:
                continue
            slots = self.model_runner.token_to_kv_pool_allocator.alloc(add_len)
            if slots is None:
                raise RuntimeError("KV cache tokens exhausted")
            start = int(self.seq_lens[row].item())
            end = start + add_len
            req_idx = int(self.req_pool_indices[row].item())
            req_tensor[req_idx, start:end] = slots.to(torch.int32)
            self.seq_lens[row] = end
            self.seq_lens_cpu[row] = end

    def _slice_slots(
        self,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        req_tensor = self.model_runner.req_to_token_pool.req_to_token
        parts = []
        for row in range(self.batch_size):
            start = int(starts[row].item())
            length = int(lengths[row].item())
            end = start + length
            parts.append(
                req_tensor[int(self.req_pool_indices[row].item()), start:end]
                .to(dtype)
                .to(self.device)
            )
        return torch.cat(parts, dim=0)

    def _global_tokens_list(self, seq_lens_cpu: torch.Tensor) -> Sequence[int]:
        return [int(val) for val in seq_lens_cpu.tolist()]

    def _base_forward_batch(
        self,
        mode: ForwardMode,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> ForwardBatch:
        forward_batch = ForwardBatch(
            forward_mode=mode,
            batch_size=self.batch_size,
            input_ids=input_ids,
            req_pool_indices=self.req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            positions=positions,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_forward_mode=mode,
            global_num_tokens_cpu=self._global_tokens_list(seq_lens_cpu),
            global_num_tokens_gpu=seq_lens.clone(),
        )

        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool
        forward_batch.attn_backend = self.model_runner.attn_backend
        return forward_batch

    def build_prefill_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> ForwardBatch:
        if not self.initialized:
            raise RuntimeError("KV cache has not been initialized")
        block_len = input_ids.size(1)
        extend = torch.full(
            (self.batch_size,), block_len, dtype=torch.int32, device=self.device
        )
        target = self.seq_lens + extend
        if torch.any(target > self.target_length):
            raise ValueError("Prefill exceeds reserved length")
        prev = self.seq_lens.clone()
        self._ensure_length(target)
        out_cache_loc = self._slice_slots(prev, extend)

        forward_batch = self._base_forward_batch(
            ForwardMode.EXTEND,
            input_ids,
            positions,
            self.seq_lens.clone(),
            self.seq_lens_cpu.clone(),
            out_cache_loc,
        )
        forward_batch.extend_prefix_lens = prev
        forward_batch.extend_prefix_lens_cpu = prev.detach().to("cpu")
        forward_batch.extend_seq_lens = extend
        forward_batch.extend_seq_lens_cpu = extend.detach().to("cpu")
        return forward_batch

    def build_decode_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> ForwardBatch:
        if not self.initialized:
            raise RuntimeError("KV cache has not been initialized")
        if block_end > self.target_length:
            raise ValueError("Decode range exceeds reserved length")

        target = torch.full(
            (self.batch_size,), block_end, dtype=torch.int32, device=self.device
        )
        self._ensure_length(target)

        starts = torch.full(
            (self.batch_size,), block_start, dtype=torch.int32, device=self.device
        )
        lengths = torch.full(
            (self.batch_size,), block_end - block_start, dtype=torch.int32, device=self.device
        )
        out_cache_loc = self._slice_slots(starts, lengths)

        return self._base_forward_batch(
            ForwardMode.DECODE,
            input_ids,
            positions,
            self.seq_lens.clone(),
            self.seq_lens_cpu.clone(),
            out_cache_loc,
        )

    def extend_cache(self, *_args, **_kwargs):
        """Compatibility shim for legacy callers."""
        return
