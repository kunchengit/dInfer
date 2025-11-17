import torch
import gc
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Callable
import tqdm
import time
from sglang.srt.utils import (
    get_available_gpu_memory,
    get_bool_env_var,
    is_hip,
)
from sglang.srt.layers.torchao_utils import save_gemlite_cache
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tp_group,
    set_custom_all_reduce,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator, graph_capture
import logging
from .utils import KVCache
import os
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.layers.dp_attention import DpPaddingMode, get_attention_tp_size
from sglang.srt.custom_op import CustomOp
from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, MHATokenToKVPool
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    CaptureHiddenMode,
)
_is_hip = is_hip()

logger = logging.getLogger(__name__)
# 假设的上下文管理器，用于在捕获期间冻结垃圾回收
@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()

    

def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with with torch.compile"""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            # Use custom-allreduce here.
            # We found the custom allreduce is much faster than the built-in allreduce in torch,
            # even with ENABLE_INTRA_NODE_COMM=1.
            # tp_group.ca_comm = None
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "default"
                ),
                dynamic=_is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm
            


# Reuse this memory pool across all cuda graph runners.
global_graph_memory_pool = None


def get_global_graph_memory_pool():
    return global_graph_memory_pool


def set_global_graph_memory_pool(val):
    global global_graph_memory_pool
    global_graph_memory_pool = val


class ModelRunner:
    def __init__(self, model: torch.nn.Module, device: str = "cuda", enable_cuda_graph: bool = True, supported_batch_sizes: Optional[list] = None, enable_compile:bool=True, server_args=None, max_length=2048, block_length=32):
        self.model = model.to(device)
        device = str(device)
        self.server_args = server_args
        if device.startswith("cuda:"):
            self.device = "cuda"
            self.gpu_id = int(device.split(":")[1])
        else:
            self.device = device
            self.gpu_id = torch.cuda.current_device()
        self.enable_compile = enable_compile
        self.enable_cuda_graph = enable_cuda_graph and (device != "cpu") # CPU 模式下禁用
        self.supported_batch_sizes = supported_batch_sizes or [1, ] # 默认支持的 batch sizes
        self.max_length = max_length
        self.block_length = block_length
        self.max_batch_size = max(self.supported_batch_sizes)
        self.page_size = 1
        self.kv_cache_dtype = torch.bfloat16
        self.req_to_token_pool: Optional[ReqToTokenPool] = None
        self.token_to_kv_pool: Optional[MHATokenToKVPool] = None
        self.token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None
        self.attn_backend: Optional[FlashAttentionBackend] = None
        self.graph_runner = None
        self.sliding_window_size = None
        self.is_hybrid = False
        self.attention_chunk_size = None
        # 设置模型为评估模式
        x = torch.arange(block_length, dtype=torch.long, device=device).unsqueeze(0)
        
        self.model.eval()
        self.tp_group = get_tp_group()
        # self.cuda_graph_runners = {}
        set_custom_all_reduce(True)

        self.model_config = getattr(self.model, "config", None)
        
        if self.server_args is not None:
            self.init_memory_pool()
            self.init_attention_backend()

        # _to_torch(self.model, reverse=True, num_tokens=1024)
        self.forward_normal(x, use_cache=True)
        # self.tp_group.ca_comm = backup_ca_comm
        if (
            self.req_to_token_pool is not None
            and self.token_to_kv_pool_allocator is not None
            and self.enable_cuda_graph
        ):
            self.init_device_graphs()
        

    def init_memory_pool(self):
        kv_dtype_str = getattr(self.server_args, "kv_cache_dtype", "auto")
        self.kv_cache_dtype = self._parse_kv_dtype(kv_dtype_str)
        enable_memory_saver = getattr(self.server_args, "enable_memory_saver", False)
        self.max_total_tokens = self.max_batch_size * self.max_length
        self.req_to_token_pool = ReqToTokenPool(
            size=self.max_batch_size,
            max_context_len=self.max_length,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
        )
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        num_kv_heads = max(1, self.model.config.num_key_value_heads // get_attention_tp_size())
        self.token_to_kv_pool = MHATokenToKVPool(
            size=self.max_total_tokens,
            page_size=self.page_size,
            dtype=self.kv_cache_dtype,
            head_num=num_kv_heads,
            head_dim=head_dim,
            layer_num=self.model.config.num_hidden_layers,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
        )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            self.max_total_tokens,
            dtype=self.kv_cache_dtype,
            device=self.device,
            kvcache=self.token_to_kv_pool,
            need_sort=False,
        )

    def init_attention_backend(self):
        if self.server_args is None:
            return

        if not hasattr(self.model_config, "context_len") or self.model_config.context_len is None:
            self.model_config.context_len = getattr(
                self.server_args, "max_model_len", self.max_length
            )

        if not hasattr(self.model_config, "attention_arch"):
            self.model_config.attention_arch = AttentionArch.MHA

        if not hasattr(self.model_config, "is_encoder_decoder"):
            self.model_config.is_encoder_decoder = False

        self.sliding_window_size = getattr(
            self.model_config, "sliding_window_size", None
        )
        self.is_hybrid = getattr(self.model_config, "is_hybrid", False)
        self.attention_chunk_size = getattr(
            self.model_config, "attention_chunk_size", None
        )
        self.attn_backend = FlashAttentionBackend(self)

    def _parse_kv_dtype(self, dtype_str: str):
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "auto": torch.bfloat16,
        }
        return mapping.get(dtype_str.lower(), torch.bfloat16)

    def alloc_kv_slots(self, batch_size: int, total_length: int):
        if self.req_to_token_pool is None or self.token_to_kv_pool_allocator is None:
            raise RuntimeError("SGLang KV cache is not initialized")
        req_indices = self.req_to_token_pool.alloc(batch_size)
        if req_indices is None:
            raise RuntimeError("No available request slots for KV cache")
        req_tensor = torch.tensor(req_indices, dtype=torch.int32, device=self.device)
        slot_rows = []
        try:
            for idx in req_indices:
                slots = self.token_to_kv_pool_allocator.alloc(total_length)
                if slots is None:
                    raise RuntimeError("KV cache memory exhausted")
                slot_rows.append(slots)
                self.req_to_token_pool.write(idx, slots.to(torch.int32))
        except Exception:
            for row in slot_rows:
                self.token_to_kv_pool_allocator.free(row)
            self.req_to_token_pool.free(req_indices)
            raise
        slot_mapping = torch.stack(slot_rows, dim=0)
        return req_tensor, slot_mapping

    def free_kv_slots(self, req_pool_indices: torch.Tensor, slot_mapping: torch.Tensor):
        if (
            self.req_to_token_pool is None
            or self.token_to_kv_pool_allocator is None
            or slot_mapping is None
        ):
            return
        for row in slot_mapping:
            self.token_to_kv_pool_allocator.free(row)
        self.req_to_token_pool.free(req_pool_indices.int().tolist())

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        self.graph_runner = CudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {'cpu graph' if self.device == 'cpu' else 'cuda graph'} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def forward_normal(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        backup_ca_comm = self.tp_group.ca_comm
        _to_torch(self.model, reverse=False, num_tokens=input_ids.numel())
        if forward_batch is not None and self.attn_backend is not None:
            forward_batch.attn_backend = self.attn_backend
            self.attn_backend.init_forward_metadata(forward_batch)
            ret = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                forward_batch=forward_batch,
                inputs_embeds=inputs_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
        else:
            ret = self.model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
                past_key_values=past_key_values,
                replace_position=replace_position,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
        self.tp_group.ca_comm = backup_ca_comm
        
        return ret
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[torch.Tensor] = None,
        past_key_values=None,
        replace_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        if isinstance(past_key_values, KVCache):
            past_key_values = past_key_values._data

        use_forward_batch = (
            forward_batch is not None
            and forward_batch.forward_mode == ForwardMode.DECODE
        )

        if (
            use_forward_batch
            and self.graph_runner
            and self.enable_cuda_graph
            and self.graph_runner.can_run(forward_batch)
        ):
            return self.graph_runner.replay(forward_batch)

        # 简化判断：如果 input_ids 的长度等于 block_length，则认为是 decode 阶段
        is_decode_phase = (
            not use_forward_batch
            and input_ids is not None
            and input_ids.shape[1] == self.block_length
            and use_cache
            and past_key_values is not None
        )
        if (
            is_decode_phase
            and self.graph_runner
            and self.enable_cuda_graph
            and self.graph_runner.can_run_legacy(input_ids, position_ids, past_key_values)
        ):
            return self.graph_runner.replay_legacy(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
        # print('run normal')
        ret = self.forward_normal(
            input_ids,
            position_ids,
            inputs_embeds,
            pp_proxy_tensors,
            past_key_values,
            replace_position,
            use_cache,
            attention_mask,
            forward_batch=forward_batch,
        )
        # if ret.past_key_values is None:
        # else:
        #     print('run normal', len(ret.past_key_values))
        # 默认路径：标准 PyTorch 执行
        return ret
    
    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
        
        
        
class CudaGraphRunner:
    """CUDA graph runner that works with SGLang ForwardBatch decode path."""

    def __init__(self, model_runner: ModelRunner):
        self.model_runner = model_runner
        self.capture_bs = sorted(set(model_runner.supported_batch_sizes))
        self.compile_bs = sorted(set(model_runner.supported_batch_sizes))
        self.device = model_runner.device
        self.device_module = torch.get_device_module(self.device)
        self.graphs = {}
        self.output_buffers = {}
        self.enable_compile = model_runner.enable_compile

        self.forward_mode = ForwardMode.DECODE

        self.max_bs = max(self.capture_bs)
        self.seq_len_fill_value = 0
        self.num_tokens_per_bs = self.model_runner.block_length

        if self.model_runner.attn_backend is not None:
            max_num_tokens = self.max_bs * self.num_tokens_per_bs
            self.model_runner.attn_backend.init_cuda_graph_state(
                self.max_bs, max_num_tokens
            )
        # self.raw_bs = 0
        # self.raw_num_token = 0
        # self.bs = 0

        with torch.device(self.device):
            total_tokens = self.max_bs * self.num_tokens_per_bs
            self.input_ids = torch.zeros(total_tokens, dtype=torch.int64, device=self.device)
            self.positions = torch.zeros(total_tokens, dtype=torch.int64, device=self.device)
            self.req_pool_indices = torch.zeros(self.max_bs, dtype=torch.int32, device=self.device)
            self.seq_lens = torch.zeros(self.max_bs, dtype=torch.int32, device=self.device)
            self.out_cache_loc = torch.zeros(total_tokens, dtype=torch.int32, device=self.device)
        self.seq_lens_cpu = torch.zeros(self.max_bs, dtype=torch.int32)
        self.forward_batches = {}

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as exc:
            raise Exception(f"Capture cuda graph failed: {exc}") from exc

    def _create_device_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with self.device_module.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _build_forward_batch(self, bs: int) -> ForwardBatch:
        num_tokens = bs * self.num_tokens_per_bs
        fb = ForwardBatch(
            forward_mode=self.forward_mode,
            batch_size=bs,
            input_ids=self.input_ids[:num_tokens].view(bs, self.num_tokens_per_bs),
            req_pool_indices=self.req_pool_indices[:bs],
            seq_lens=self.seq_lens[:bs],
            seq_lens_cpu=self.seq_lens_cpu[:bs],
            out_cache_loc=self.out_cache_loc[:num_tokens],
            seq_lens_sum=bs * self.num_tokens_per_bs,
            positions=self.positions[:num_tokens].view(bs, self.num_tokens_per_bs),
            capture_hidden_mode=CaptureHiddenMode.NULL,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_forward_mode=self.forward_mode,
            global_num_tokens_cpu=[self.num_tokens_per_bs] * bs,
            global_num_tokens_gpu=self.seq_lens[:bs],
        )
        fb.req_to_token_pool = self.model_runner.req_to_token_pool
        fb.token_to_kv_pool = self.model_runner.token_to_kv_pool
        fb.attn_backend = self.model_runner.attn_backend
        return fb

    def capture(self) -> None:
        with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc), graph_capture() as graph_capture_context:
            self.stream = graph_capture_context.stream
            capture_range = (
                tqdm.tqdm(list(reversed(self.capture_bs)))
                if get_tensor_model_parallel_rank() == 0
                else reversed(self.capture_bs)
            )
            for bs in capture_range:
                with patch_model(
                    self.model_runner.model,
                    bs in self.compile_bs and self.enable_compile,
                    num_tokens=bs * self.num_tokens_per_bs,
                    tp_group=self.model_runner.tp_group,
                ) as forward:
                    graph, output = self.capture_one_batch_size(bs, forward)
                    self.graphs[bs] = graph
                    self.output_buffers[bs] = output

                # Save gemlite cache after each capture
                save_gemlite_cache()

    def capture_one_batch_size(self, bs: int, forward: Callable):
        graph = self._create_device_graph()
        stream = self.stream
        num_tokens = bs * self.num_tokens_per_bs

        forward_batch = self._build_forward_batch(bs)
        self.seq_lens[:bs].fill_(self.num_tokens_per_bs)
        self.seq_lens_cpu[:bs].fill_(self.num_tokens_per_bs)
        self.req_pool_indices[:bs].copy_(torch.arange(bs, dtype=torch.int32, device=self.device))
        self.positions[:num_tokens] = torch.arange(self.num_tokens_per_bs, device=self.device).repeat(bs)
        self.input_ids[:num_tokens].zero_()
        self.out_cache_loc[:num_tokens].zero_()
        forward_batch.seq_lens_sum = int(self.seq_lens[:bs].sum().item())
        prefix = torch.zeros(bs, dtype=torch.int32, device=self.device)
        extend = torch.full(
            (bs,), self.num_tokens_per_bs, dtype=torch.int32, device=self.device
        )
        forward_batch.extend_prefix_lens = prefix
        forward_batch.extend_prefix_lens_cpu = prefix.detach().to("cpu")
        forward_batch.extend_seq_lens = extend
        forward_batch.extend_seq_lens_cpu = extend.detach().to("cpu")
        forward_batch.is_diffusion = True

        self.model_runner.attn_backend.init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            None,
            self.forward_mode,
            None,
        )

        def run_once():
            return forward(
                input_ids=forward_batch.input_ids,
                position_ids=forward_batch.positions,
                inputs_embeds=None,
                pp_proxy_tensors=None,
                past_key_values=None,
                replace_position=(0, 0),
                use_cache=True,
                attention_mask=None,
                forward_batch=forward_batch,
            )

        for _ in range(2):
            self.device_module.synchronize()
            self.model_runner.tp_group.barrier()
            run_once()

        if get_global_graph_memory_pool() is None:
            set_global_graph_memory_pool(self.device_module.graph_pool_handle())
        set_graph_pool_id(get_global_graph_memory_pool())
        out = self._capture_graph(graph, get_global_graph_memory_pool(), stream, run_once)

        self.forward_batches[bs] = forward_batch
        return graph, out

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        return (
            forward_batch.forward_mode == self.forward_mode
            and forward_batch.batch_size in self.graphs
        )

    def replay_prepare(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if bs not in self.graphs:
            raise ValueError(f"Unsupported CUDA graph batch size {bs}")

        num_tokens = bs * self.num_tokens_per_bs
        self.input_ids[:num_tokens].copy_(forward_batch.input_ids.reshape(-1))
        self.positions[:num_tokens].copy_(forward_batch.positions.reshape(-1))
        self.req_pool_indices[:bs].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:bs].copy_(forward_batch.seq_lens)
        self.seq_lens_cpu[:bs].copy_(forward_batch.seq_lens_cpu)
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc.reshape(-1))

        seq_lens_sum = int(forward_batch.seq_lens_sum)
        self.model_runner.attn_backend.init_forward_metadata_replay_cuda_graph(
            bs,
            self.req_pool_indices[:bs],
            self.seq_lens[:bs],
            seq_lens_sum,
            None,
            self.forward_mode,
            None,
            seq_lens_cpu=self.seq_lens_cpu[:bs],
            out_cache_loc=self.out_cache_loc[:num_tokens],
        )

        self.raw_bs = bs
        self.raw_num_token = num_tokens
        self.bs = bs

    def replay(self, forward_batch: ForwardBatch):
        self.replay_prepare(forward_batch)
        self.graphs[self.bs].replay()
        return self.output_buffers[self.bs]

    def can_run_legacy(self, *args, **kwargs) -> bool:
        return False

    def replay_legacy(self, *args, **kwargs):
        raise RuntimeError("Legacy CUDA graph path is not available with the SGLang backend.")
