from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool


@dataclass
class CacheInfoSglang:
    token_to_kv_pool: MHATokenToKVPool
    cache_loc: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int


class BlockDiffusionSGLangCacheManager:
    """Thin wrapper around SGLang KV cache primitives for offline block diffusion."""

    supports_sglang = True

    def __init__(self, model_config, server_args=None, dtype=torch.bfloat16):
        self.model_config = model_config
        self.server_args = server_args
        self.dtype = dtype
        self.initialized = False
        self.req_to_token_pool: Optional[ReqToTokenPool] = None
        self.req_pool_indices: Optional[torch.Tensor] = None
        self.token_to_kv_pool: Optional[MHATokenToKVPool] = None
        self.token_allocator: Optional[TokenToKVPoolAllocator] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self.batch_size: Optional[int] = None
        self.max_context_len: Optional[int] = None
        self.device: Optional[torch.device] = None

    def initialize(self, batch_size: int, total_length: int, device: torch.device) -> None:
        if self.initialized:
            return
        tp_size = max(get_attention_tp_size(), 1)
        kv_heads = max(self.model_config.num_key_value_heads // tp_size, 1)
        head_dim = (
            self.model_config.head_dim
            if hasattr(self.model_config, "head_dim")
            else self.model_config.hidden_size // self.model_config.num_attention_heads
        )
        layer_num = getattr(self.model_config, "num_hidden_layers", len(self.model_config.layers))
        max_total_tokens = batch_size * total_length

        self.req_to_token_pool = ReqToTokenPool(
            batch_size,
            total_length,
            device=str(device),
            enable_memory_saver=False,
        )
        alloc_indices = self.req_to_token_pool.alloc(batch_size)
        self.req_pool_indices = torch.tensor(
            alloc_indices, dtype=torch.int64, device=device
        )
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_tokens,
            page_size=1,
            dtype=self.dtype,
            head_num=kv_heads,
            head_dim=head_dim,
            layer_num=layer_num,
            device=str(device),
            enable_memory_saver=False,
            start_layer=0,
            end_layer=layer_num - 1,
            enable_alt_stream=False,
            enable_kv_cache_copy=False,
        )
        self.token_allocator = TokenToKVPoolAllocator(
            max_total_tokens,
            dtype=torch.int64,
            device=str(device),
            kvcache=self.token_to_kv_pool,
            need_sort=False,
        )
        self.seq_lens = torch.zeros(batch_size, dtype=torch.int64, device=device)
        self.batch_size = batch_size
        self.max_context_len = total_length
        self.device = device
        self.initialized = True

    def reserve_block(self, block_len: int) -> CacheInfoSglang:
        if not self.initialized:
            raise RuntimeError("BlockDiffusionSGLangCacheManager must be initialized before use.")
        if block_len <= 0:
            raise ValueError("block_len must be positive.")
        projected = self.seq_lens.max().item() + block_len
        if projected > self.max_context_len:
            raise RuntimeError("KV cache exhausted: context length exceeded.")
        cache_loc = self._allocate_block(block_len)
        max_seq_len = int(projected)
        page_table = self.req_to_token_pool.req_to_token.index_select(
            0, self.req_pool_indices.long()
        )[:, :max_seq_len].to(device=self.device)
        return CacheInfoSglang(
            token_to_kv_pool=self.token_to_kv_pool,
            cache_loc=cache_loc,
            page_table=page_table.contiguous(),
            seq_lens=self.seq_lens.clone(),
            max_seq_len=max_seq_len,
        )

    def _allocate_block(self, block_len: int) -> torch.Tensor:
        total = block_len * self.batch_size
        loc = self.token_allocator.alloc(int(total))
        if loc is None:
            raise RuntimeError("Failed to allocate KV cache slots.")
        loc = loc.view(self.batch_size, block_len)
        for idx in range(self.batch_size):
            start = int(self.seq_lens[idx].item())
            end = start + block_len
            self.req_to_token_pool.req_to_token[
                self.req_pool_indices[idx], start:end
            ] = loc[idx]
        self.seq_lens += block_len
        return loc
