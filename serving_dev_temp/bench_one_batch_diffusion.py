"""
Benchmark utilities for DiffusionWorker / DiffusionIteration without launching a server.

The script mirrors ``bench_one_batch.py`` but drives the upcoming diffusion
execution path. It is meant for two purposes:

1. Correctness sanity check against the existing ModelRunner pipeline.
2. Latency benchmarking of a single static batch (prefill + decode) using
   diffusion decoding kernels.

Usage examples (placeholders until DiffusionWorker is fully implemented):

```
# Correctness check with baseline comparison
python -m sglang.bench_one_batch_diffusion --model-path meta-llama/Meta-Llama-3-8B-Instruct --correct

# Latency sweep
python -m sglang.bench_one_batch_diffusion --model-path meta-llama/Meta-Llama-3-8B-Instruct --batch-size 1 4 --input-len 256 512 --output-len 32

# Skip baseline comparison to save memory (latency only)
python -m sglang.bench_one_batch_diffusion --model-path meta-llama/Meta-Llama-3-8B-Instruct --skip-baseline
```
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import logging
import multiprocessing
import os
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import (
    configure_logger,
    is_cuda_alike,
    is_xpu,
    require_mlp_sync,
    require_mlp_tp_gather,
    suppress_other_loggers,
)
from sglang.srt.utils.hf_transformers_utils import get_tokenizer

try:
    from sglang.srt.diffusion.worker import DiffusionWorker
except ImportError as exc:  # pragma: no cover - pending diffusion implementation
    DiffusionWorker = None  # type: ignore[assignment]
    _DIFFUSION_IMPORT_ERROR = exc
else:
    _DIFFUSION_IMPORT_ERROR = None

profile_activities = [torch.profiler.ProfilerActivity.CPU] + [
    profiler_activity
    for available, profiler_activity in [
        (is_cuda_alike(), torch.profiler.ProfilerActivity.CUDA),
        (is_xpu(), torch.profiler.ProfilerActivity.XPU),
    ]
    if available
]


@dataclasses.dataclass
class DiffusionBenchArgs:
    """CLI arguments shared by correctness and latency benchmarks."""

    run_name: str = "diffusion_default"
    batch_size: Tuple[int, ...] = (1,)
    input_len: Tuple[int, ...] = (1024,)
    output_len: Tuple[int, ...] = (16,)
    prompt_filename: str = ""
    result_filename: str = "diffusion_result.jsonl"
    correctness_test: bool = False
    cut_len: int = 4
    log_decode_step: int = 0
    profile: bool = False
    profile_record_shapes: bool = False
    profile_filename_prefix: str = "diffusion_profile"
    block_size: int = 64
    skip_baseline: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--run-name", type=str, default=DiffusionBenchArgs.run_name)
        parser.add_argument(
            "--batch-size", type=int, nargs="+", default=list(DiffusionBenchArgs.batch_size)
        )
        parser.add_argument(
            "--input-len", type=int, nargs="+", default=list(DiffusionBenchArgs.input_len)
        )
        parser.add_argument(
            "--output-len", type=int, nargs="+", default=list(DiffusionBenchArgs.output_len)
        )
        parser.add_argument(
            "--prompt-filename", type=str, default=DiffusionBenchArgs.prompt_filename
        )
        parser.add_argument(
            "--result-filename", type=str, default=DiffusionBenchArgs.result_filename
        )
        parser.add_argument(
            "--correct",
            dest="correctness_test",
            action="store_true",
            help="Run correctness test instead of latency sweep.",
        )
        parser.add_argument("--cut-len", type=int, default=DiffusionBenchArgs.cut_len)
        parser.add_argument(
            "--log-decode-step",
            type=int,
            default=DiffusionBenchArgs.log_decode_step,
            help="Print decode latency every N steps (0 disables logging).",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Enable torch.profiler during decode stage.",
        )
        parser.add_argument(
            "--profile-record-shapes",
            action="store_true",
            help="Record shapes when profiling decode stage.",
        )
        parser.add_argument(
            "--profile-filename-prefix",
            type=str,
            default=DiffusionBenchArgs.profile_filename_prefix,
        )
        parser.add_argument(
            "--block-size",
            type=int,
            default=DiffusionBenchArgs.block_size,
            help="Diffusion block size (number of masked tokens per extend).",
        )
        parser.add_argument(
            "--skip-baseline",
            action="store_true",
            help="Skip baseline ModelRunner comparison to save memory.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "DiffusionBenchArgs":
        return cls(
            run_name=args.run_name,
            batch_size=tuple(args.batch_size),
            input_len=tuple(args.input_len),
            output_len=tuple(args.output_len),
            prompt_filename=args.prompt_filename,
            result_filename=args.result_filename,
            correctness_test=args.correctness_test,
            cut_len=args.cut_len,
            log_decode_step=args.log_decode_step,
            profile=args.profile,
            profile_record_shapes=args.profile_record_shapes,
            profile_filename_prefix=args.profile_filename_prefix,
            block_size=args.block_size,
            skip_baseline=args.skip_baseline,
        )


def _ensure_diffusion_worker_available() -> None:
    if DiffusionWorker is None:
        raise ImportError(
            "DiffusionWorker is not available. Implement sglang.srt.diffusion.worker "
            "before running diffusion benchmarks."
        ) from _DIFFUSION_IMPORT_ERROR


def load_model(server_args: ServerArgs, port_args: PortArgs, tp_rank: int) -> Tuple[ModelRunner, torch.nn.Module]:
    suppress_other_loggers()
    moe_ep_rank = tp_rank // max(1, (server_args.tp_size // server_args.ep_size))

    model_config = ModelConfig.from_server_args(server_args)
    model_runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return model_runner, tokenizer


def load_diffusion_worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    tp_rank: int,
    block_size: int,
) -> DiffusionWorker:
    _ensure_diffusion_worker_available()
    suppress_other_loggers()
    moe_ep_rank = tp_rank // max(1, (server_args.tp_size // server_args.ep_size))
    model_config = ModelConfig.from_server_args(server_args)
    worker = DiffusionWorker(
        server_args=server_args,
        model_config=model_config,
        gpu_id=tp_rank,
        tp_rank=tp_rank,
        pp_rank=0,
        moe_ep_rank=moe_ep_rank,
        graph_runner=None,
        enable_cuda_graph_sampling=not server_args.disable_cuda_graph,
        block_size=block_size,
        nccl_port=port_args.nccl_port,
    )
    if server_args.tp_size > 1:
        dist.barrier()
    return worker


def prepare_inputs_for_correctness_test(
    bench_args: DiffusionBenchArgs,
    tokenizer,
    custom_prompts: Optional[List[str]] = None,
) -> Tuple[List[List[int]], List[Req]]:
    prompts = (
        custom_prompts
        if custom_prompts
        else [
            "The capital of France is",
            "The capital of the United Kindom is",
            "Today is a sunny day and I like",
        ]
    )
    input_ids = [tokenizer.encode(p) for p in prompts]
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=bench_args.output_len[0],
    )

    reqs: List[Req] = []
    for i, prompt_ids in enumerate(input_ids):
        if len(prompt_ids) <= bench_args.cut_len:
            raise ValueError(
                f"Prompt {i} is shorter than cut_len={bench_args.cut_len}; "
                "increase prompt length or decrease cut_len."
            )
        truncated = prompt_ids[: bench_args.cut_len]
        req = Req(
            rid=i,
            origin_input_text=prompts[i],
            origin_input_ids=truncated,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    return input_ids, reqs


def prepare_extend_inputs_for_correctness_test(
    bench_args: DiffusionBenchArgs,
    input_ids: List[List[int]],
    reqs: List[Req],
    pool_owner,
) -> List[Req]:
    for i, req in enumerate(reqs):
        req.fill_ids += input_ids[i][bench_args.cut_len :]
        req.prefix_indices = pool_owner.req_to_token_pool.req_to_token[
            i, : bench_args.cut_len
        ]
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
    return reqs


def prepare_synthetic_inputs_for_latency_test(
    bench_args: DiffusionBenchArgs,
    batch_size: int,
    input_len: int,
    custom_inputs: Optional[List[List[int]]] = None,
) -> List[Req]:
    input_ids = (
        custom_inputs
        if custom_inputs
        else np.random.randint(
            0, 10000, (batch_size, input_len), dtype=np.int32
        ).tolist()
    )
    sampling_params = SamplingParams(
        temperature=0,
        max_new_tokens=max(bench_args.output_len),
    )

    reqs: List[Req] = []
    for i, ids in enumerate(input_ids):
        req = Req(
            rid=i,
            origin_input_text="",
            origin_input_ids=list(ids),
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
        req.logprob_start_len = len(req.origin_input_ids) - 1
        reqs.append(req)
    return reqs


def _build_dummy_tree_cache(runtime) -> SimpleNamespace:
    return SimpleNamespace(
        page_size=1,
        device=runtime.device,
        token_to_kv_pool_allocator=runtime.token_to_kv_pool_allocator,
    )


def _maybe_prepare_mlp_sync_batch(batch: ScheduleBatch, runtime) -> None:
    if require_mlp_sync(runtime.server_args):
        Scheduler.prepare_mlp_sync_batch_raw(
            batch,
            dp_size=runtime.server_args.dp_size,
            attn_tp_size=1,
            tp_group=getattr(runtime, "tp_group", None),
            get_idle_batch=None,
            disable_cuda_graph=runtime.server_args.disable_cuda_graph,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            speculative_num_draft_tokens=None,
            require_mlp_tp_gather=require_mlp_tp_gather(runtime.server_args),
            disable_overlap_schedule=runtime.server_args.disable_overlap_schedule,
        )


@torch.no_grad()
def diffusion_prefill(
    reqs: List[Req],
    worker: DiffusionWorker,
) -> Tuple["GenerationBatchResult", object, ScheduleBatch]:
    dummy_tree_cache = _build_dummy_tree_cache(worker)
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=worker.req_to_token_pool,
        token_to_kv_pool_allocator=worker.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=worker.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, worker)
    model_worker_batch = batch.get_model_worker_batch()
    output = worker.prefill(model_worker_batch)
    if isinstance(output, tuple):
        generation_result, batch_state = output
    else:
        generation_result = output
        batch_state = getattr(generation_result, "diffusion_state", None)
    return generation_result, batch_state, batch


@torch.no_grad()
def diffusion_decode(
    next_token_ids: torch.Tensor,
    batch: ScheduleBatch,
    worker: DiffusionWorker,
    batch_state: object,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], object]:
    batch.output_ids = next_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, worker)
    model_worker_batch = batch.get_model_worker_batch()
    output = worker.extend(model_worker_batch, batch_state)
    if isinstance(output, tuple):
        generation_result, batch_state = output
    else:
        generation_result = output
        batch_state = getattr(generation_result, "diffusion_state", None)
    logits = getattr(generation_result, "next_token_logits", None)
    return generation_result.next_token_ids, logits, batch_state


def synchronize(device: torch.device) -> None:
    torch.get_device_module(device).synchronize()


def _format_latency_result(
    run_name: str,
    model_path: str,
    batch_size: int,
    input_len: int,
    output_len: int,
    prefill_latency: float,
    decode_latency: float,
    total_latency: float,
    prefill_tokens: int,
    decode_tokens: int,
) -> dict:
    return {
        "model_path": model_path,
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "prefill_latency_ms": prefill_latency * 1000,
        "decode_latency_ms": decode_latency * 1000,
        "total_latency_ms": total_latency * 1000,
        "prefill_throughput_tps": (prefill_tokens / prefill_latency)
        if prefill_latency > 0
        else 0.0,
        "decode_throughput_tps": (decode_tokens / decode_latency)
        if decode_latency > 0
        else 0.0,
        "overall_throughput_tps": (decode_tokens / total_latency)
        if total_latency > 0
        else 0.0,
    }


def diffusion_correctness_test(
    server_args: ServerArgs,
    port_args: PortArgs,
    bench_args: DiffusionBenchArgs,
    tp_rank: int,
) -> None:
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else (lambda *args, **kwargs: None)

    diffusion_worker = load_diffusion_worker(
        server_args, port_args, tp_rank, bench_args.block_size
    )

    baseline_runner: Optional[ModelRunner] = None
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )
    if not bench_args.skip_baseline:
        baseline_runner, tokenizer = load_model(server_args, port_args, tp_rank)

    custom_prompts = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    input_ids, reqs = prepare_inputs_for_correctness_test(
        bench_args, tokenizer, custom_prompts
    )
    rank_print(f"\ninput_ids={input_ids}\n")

    if baseline_runner is not None:
        baseline_reqs = copy.deepcopy(reqs)
        if bench_args.cut_len > 0:
            base_next_ids, base_logits, base_batch = extend(baseline_reqs, baseline_runner)
            rank_print(f"baseline prefill logits (first half): {base_logits}\n")
        baseline_reqs = prepare_extend_inputs_for_correctness_test(
            bench_args, input_ids, baseline_reqs, baseline_runner
        )
        base_next_ids, base_logits_final, base_batch = extend(baseline_reqs, baseline_runner)
        rank_print(f"baseline prefill logits (final): {base_logits_final}\n")
        baseline_outputs = [
            input_ids[i] + [base_next_ids[i].item()] for i in range(len(input_ids))
        ]
        for _ in range(bench_args.output_len[0] - 1):
            base_next_ids, _ = decode(base_next_ids, base_batch, baseline_runner)
            ids_list = base_next_ids.tolist()
            for i, tid in enumerate(ids_list):
                baseline_outputs[i].append(tid)
    else:
        baseline_outputs = None

    diffusion_reqs = copy.deepcopy(reqs)
    if bench_args.cut_len > 0:
        diff_prefill_result, diff_state, diff_batch = diffusion_prefill(diffusion_reqs, diffusion_worker)
        diff_logits = getattr(diff_prefill_result, "next_token_logits", None)
        rank_print(f"diffusion prefill logits (first half): {diff_logits}\n")
        next_token_ids = diff_prefill_result.next_token_ids
    else:
        diff_state = None
        diff_batch = None
        next_token_ids = None

    diffusion_reqs = prepare_extend_inputs_for_correctness_test(
        bench_args, input_ids, diffusion_reqs, diffusion_worker
    )
    diff_prefill_result, diff_state, diff_batch = diffusion_prefill(diffusion_reqs, diffusion_worker)
    diff_logits_final = getattr(diff_prefill_result, "next_token_logits", None)
    rank_print(f"diffusion prefill logits (final): {diff_logits_final}\n")
    next_token_ids = diff_prefill_result.next_token_ids

    diffusion_outputs = [
        input_ids[i] + [next_token_ids[i].item()] for i in range(len(input_ids))
    ]
    for step in range(bench_args.output_len[0] - 1):
        next_token_ids, _, diff_state = diffusion_decode(
            next_token_ids, diff_batch, diffusion_worker, diff_state
        )
        ids_list = next_token_ids.tolist()
        for i, tid in enumerate(ids_list):
            diffusion_outputs[i].append(tid)
        if bench_args.log_decode_step and (step + 1) % bench_args.log_decode_step == 0:
            rank_print(f"[Diffusion] step {step + 1}: tokens={ids_list}")

    for i, tokens in enumerate(diffusion_outputs):
        rank_print(f"========== Diffusion Prompt {i} ==========")
        rank_print(tokenizer.decode(tokens), "\n")

    if baseline_outputs is not None:
        total_diff = 0
        for base, diff in zip(baseline_outputs, diffusion_outputs):
            if base != diff:
                total_diff += 1
        if total_diff == 0:
            rank_print("Diffusion outputs match baseline ModelRunner outputs.")
        else:
            rank_print(f"WARNING: {total_diff} prompts diverged from baseline outputs.")


@torch.no_grad()
def extend(reqs: List[Req], model_runner: ModelRunner):
    dummy_tree_cache = SimpleNamespace(
        page_size=1,
        device=model_runner.device,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=dummy_tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits, batch


@torch.no_grad()
def decode(next_token_ids: torch.Tensor, batch: ScheduleBatch, model_runner: ModelRunner):
    batch.output_ids = next_token_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    model_worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(model_worker_batch, model_runner)
    logits_output, _ = model_runner.forward(forward_batch)
    next_token_ids = model_runner.sample(logits_output, forward_batch)
    return next_token_ids, logits_output.next_token_logits


def _read_prompts_from_file(prompt_file: str, rank_print) -> List[str]:
    if not prompt_file:
        return []
    try:
        with open(prompt_file, "r", encoding="utf-8") as pf:
            return pf.readlines()
    except FileNotFoundError:
        rank_print(f"Custom prompt file {prompt_file} not found. Using default prompts.")
        return []


def _save_profile_trace_results(profiler, filename: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent_dir, exist_ok=True)
    profiler.export_chrome_trace(filename)
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total"
        )
    )


def diffusion_latency_test_run_once(
    run_name: str,
    server_args: ServerArgs,
    worker: DiffusionWorker,
    reqs: List[Req],
    batch_size: int,
    input_len: int,
    output_len: int,
    device: torch.device,
    log_decode_step: int,
    profile: Optional[bool],
    profile_record_shapes: Optional[bool],
    profile_filename_prefix: str,
) -> Optional[dict]:
    synchronize(device)
    prefill_tokens = sum(len(req.fill_ids) for req in reqs)
    start = time.perf_counter()
    prefill_result, batch_state, batch = diffusion_prefill(reqs, worker)
    synchronize(device)
    prefill_latency = time.perf_counter() - start

    if not hasattr(prefill_result, "next_token_ids") or prefill_result.next_token_ids is None:
        return None

    next_token_ids = prefill_result.next_token_ids
    decode_tokens = batch_size * output_len
    decode_latency = 0.0
    total_latency = prefill_latency

    profiler_ctx = (
        torch.profiler.profile(
            activities=profile_activities,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=lambda prof: _save_profile_trace_results(
                prof, f"{profile_filename_prefix}_diffusion_decode.json"
            ),
            record_shapes=bool(profile_record_shapes),
            profile_memory=False,
        )
        if profile
        else nullcontext()
    )

    with profiler_ctx as prof:
        for step in range(output_len):
            synchronize(device)
            start_decode = time.perf_counter()
            next_token_ids, _, batch_state = diffusion_decode(
                next_token_ids, batch, worker, batch_state
            )
            synchronize(device)
            step_latency = time.perf_counter() - start_decode
            decode_latency += step_latency
            total_latency += step_latency
            if log_decode_step and (step + 1) % log_decode_step == 0:
                print(f"[Diffusion] decode step {step + 1}: {step_latency * 1000:.3f} ms")
            if profile:
                assert prof is not None
                prof.step()

    return _format_latency_result(
        run_name,
        server_args.model_path,
        batch_size,
        input_len,
        output_len,
        prefill_latency,
        decode_latency,
        total_latency,
        prefill_tokens,
        decode_tokens,
    )


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *excinfo):
        return False


def diffusion_latency_test(
    server_args: ServerArgs,
    port_args: PortArgs,
    bench_args: DiffusionBenchArgs,
    tp_rank: int,
) -> None:
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else (lambda *args, **kwargs: None)

    diffusion_worker = load_diffusion_worker(
        server_args, port_args, tp_rank, bench_args.block_size
    )
    tokenizer = get_tokenizer(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )

    # warmup
    warmup_reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        None,
    )
    diffusion_latency_test_run_once(
        bench_args.run_name,
        server_args,
        diffusion_worker,
        warmup_reqs,
        bench_args.batch_size[0],
        bench_args.input_len[0],
        min(32, bench_args.output_len[0]),
        diffusion_worker.device,
        0,
        None,
        None,
        "",
    )

    rank_print("Benchmark DiffusionWorker ...")

    custom_inputs = _read_prompts_from_file(bench_args.prompt_filename, rank_print)
    custom_inputs = [tokenizer.encode(p.strip()) for p in custom_inputs]
    custom_len = len(custom_inputs)

    results = []
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        aligned_inputs: Optional[List[List[int]]] = None
        if custom_inputs:
            if custom_len == bs:
                aligned_inputs = copy.deepcopy(custom_inputs)
            elif custom_len > bs:
                rank_print(
                    f"Custom input size ({custom_len}) > batch_size ({bs}); using first {bs} prompts."
                )
                aligned_inputs = copy.deepcopy(custom_inputs[:bs])
            else:
                rank_print(
                    f"Custom input size ({custom_len}) < batch_size ({bs}); padding with last prompt."
                )
                aligned_inputs = copy.deepcopy(custom_inputs)
                aligned_inputs.extend([aligned_inputs[-1]] * (bs - custom_len))

        reqs = prepare_synthetic_inputs_for_latency_test(
            bench_args, bs, il, aligned_inputs
        )
        ret = diffusion_latency_test_run_once(
            bench_args.run_name,
            server_args,
            diffusion_worker,
            reqs,
            bs,
            il,
            ol,
            diffusion_worker.device,
            bench_args.log_decode_step,
            bench_args.profile if tp_rank == 0 else None,
            bench_args.profile_record_shapes if tp_rank == 0 else None,
            bench_args.profile_filename_prefix,
        )
        if ret is not None:
            results.append(ret)

    if tp_rank == 0 and bench_args.result_filename:
        with open(bench_args.result_filename, "a", encoding="utf-8") as fout:
            for result in results:
                fout.write(json.dumps(result) + "\n")

    if server_args.tp_size > 1:
        destroy_distributed_environment()


def main(server_args: ServerArgs, bench_args: DiffusionBenchArgs) -> None:
    _set_envs_and_config(server_args)

    if server_args.model_path is None:
        raise ValueError("Provide --model-path to benchmark diffusion worker.")

    initialize_moe_config(server_args)
    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        if bench_args.correctness_test:
            diffusion_correctness_test(server_args, port_args, bench_args, 0)
        else:
            diffusion_latency_test(server_args, port_args, bench_args, 0)
    else:
        work_func = (
            diffusion_correctness_test if bench_args.correctness_test else diffusion_latency_test
        )
        workers = []
        for tp_rank in range(server_args.tp_size):
            proc = multiprocessing.Process(
                target=work_func,
                args=(
                    server_args,
                    port_args,
                    bench_args,
                    tp_rank,
                ),
            )
            proc.start()
            workers.append(proc)
        for proc in workers:
            proc.join()
        for proc in workers:
            proc.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    DiffusionBenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = DiffusionBenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        main(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(os.getpid(), include_parent=False)
