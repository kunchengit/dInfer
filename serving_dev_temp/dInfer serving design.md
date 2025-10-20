# dInfer-SGLang 扩散推理设计

## sglang 原生 prefill/extend 推理流程

### 调用关系示意
```
HTTP 请求
  |
  v
Scheduler.event_loop 收到 TokenizedGenerateReqInput
  |
  v
创建 Req（RequestStage.PREFILL）并进入待调度队列
  |
  v
PrefillAdder 聚合请求 -> ScheduleBatch(ForwardMode.PREFILL)
  |
  v
序列化为 ModelWorkerBatch 发送给 TpModelWorker
  |
  v
TpModelWorker.forward
    -> ForwardBatch.init_new
    -> ModelRunner.forward
    -> ModelRunner.sample
  |
  v
返回 GenerationBatchResult 给 Scheduler
  |
  v
Scheduler.process_batch_result
    -> 更新 Req 状态，如果未完成则转入 RequestStage.DECODE
  |
  v
Decode 队列触发 ExtendAdder -> ScheduleBatch(ForwardMode.DECODE)
  |
  v
ModelWorkerBatch (decode) 下发至 TpModelWorker
  |
  v
TpModelWorker.forward (decode)
    -> ForwardBatch.init_from_extend
    -> ModelRunner.forward
    -> ModelRunner.sample
  |
  v
GenerationBatchResult 回传
  |
  v
Scheduler.process_batch_result
    -> 写回下一个 token，判断是否继续 extend 或结束请求
```

### 第一步：请求接入与 Req 构建
1. `Scheduler` 通过 ZeroMQ 收到 `TokenizedGenerateReqInput` 或 `BatchTokenizedGenerateReqInput`。
2. `Scheduler.handle_generate_req` 调用 `add_req`：
   - 创建 `Req` 对象（位于 `python/sglang/srt/managers/schedule_batch.py`），记录 `rid`、`prompt_token_ids`、`sampling_params`、`max_new_tokens`、`RequestStage.PREFILL` 等。
   - 将 `Req` 放入 `self.reqs` 字典，并把 `prefill_position = 0`，`decode_position = 0`。
   - 如果启用分块 kv 管理，会在此阶段调用 `token_to_kv_pool_allocator.reserve` 预估容量。

### 第二步：prefill 批次构建与发送
1. `Scheduler.event_loop` 周期性运行 `maybe_schedule_prefill`：
   - `PrefillAdder` 检查等待中的 `Req`，依据并行度限制、`prefill_batch_size`、`max_total_num_tokens` 聚合成 `ScheduleBatch`。
   - 若涉及 `ReqToTokenPool`，此时分配 `req_pool_indices`，并登记在 `ScheduleBatch.out_cache_loc`。
2. `ScheduleBatch`（`ForwardMode.PREFILL`）关键字段：
   - `reqs`：参与批次的 `Req` 列表。
   - `input_ids`：拼接后的 prompt token。
   - `seq_lens_cpu`：各序列长度。
   - `sampling_info`：`SamplingBatchInfo`，含 `temperature`、`top_p`、`top_k`、正则约束等。
3. `Scheduler._send_to_worker` 将 `ScheduleBatch` 转换为 `ModelWorkerBatch`：
   - 复制 `input_ids` 到 GPU。
   - 构造 `req_pool_indices`、`out_cache_loc`、`seq_lens`。
   - `forward_mode` 设为 `ForwardMode.PREFILL`。
4. 通过 ZeroMQ PUSH 套接字发送 `ModelWorkerBatch` 到指定的 `TpModelWorker` 进程。

### 第三步：TpModelWorker 执行 prefill
1. `TpModelWorker.forward` 收到 `ModelWorkerBatch`：
   - 调用 `ForwardBatch.init_new(model_worker_batch, self.model_runner)`。
   - `ForwardBatch` 在 GPU 上准备好 `positions`、`seq_lens`、`sampling_info`、`input_ids`。
2. `ModelRunner.forward` 执行模型前向：
   - 若首次，初始化 `CudaGraphRunner`、注意力算子、MoE 分布。
   - 前向结果封装在 `LogitsProcessorOutput`。
3. 若 `model_worker_batch.return_logprob` 为真，调用 `ModelRunner.compute_logprobs_only`；否则通过 `ModelRunner.sample` 采样下一个 token：
   - 使用 `Sampler`、`LogitsProcessor`、正则约束等组件。
4. `TpModelWorker` 将结果封装进 `GenerationBatchResult`：
   - `next_token_ids`：prefill 仅当 `is_prefill_only`。
   - `delay_sample_func`：必要时延迟采样。
   - `pp_hidden_states_proxy_tensors`：流水线并行时传递中间态。
   - `can_run_cuda_graph`：用于后续缓存 graph。
5. 通过 ZeroMQ 返回 `GenerationBatchResult` 和批次元数据。

### 第四步：Scheduler 处理 prefill 结果
1. `Scheduler.process_batch_result`：
   - 将 `GenerationBatchResult.next_token_ids` 写回每个 `Req`，更新 `Req.output_token_ids`。
   - `Req.prefill_position` 更新为 prompt 长度。
   - `token_to_kv_pool_allocator.bind`：将 `GenerationBatchResult` 的缓存句柄和 `Req` 关联，便于 decode 使用。
2. 若请求设置了 `is_prefill_only` 或者已经命中停止条件，`Scheduler.finish_req` 直接返回响应。
3. 否则把请求阶段改为 `RequestStage.DECODE`，加入解码优先队列。此时 `Req.extend_length = 1`（按 token）或根据策略调整。

### 第五步：extend 批次构建
1. `Scheduler.maybe_schedule_decode` 利用 `ExtendAdder` 选择可以继续解码的 `Req`：
   - 根据 `max_num_batched_tokens_decode`、`spec_decode_parallelism` 等限制选择候选。
   - 通过 `alloc_for_decode` 或 `alloc_for_extend` 从 `token_to_kv_pool_allocator` 分配写入槽位，更新 `ScheduleBatch.out_cache_loc`。
2. `ScheduleBatch`（`ForwardMode.DECODE`）特定字段：
   - `extend_num_tokens`：本批次将尝试生成的 token 数，一般为各请求 1。
   - `extend_prefix_lens`：每个请求的 prompt 长度，用于定位 KV。
   - `extend_logprob_start_lens`：记录 logprob 写入起点。
   - `is_extend_in_batch`、`global_num_tokens`：面向 DP attention。
3. `Scheduler._send_to_worker` 构造 `ModelWorkerBatch`：
   - `input_ids`：包含 `last_token` 或特殊 mask。
   - `req_pool_indices` 和 `out_cache_loc` 指向 KV 槽。
   - `forward_mode` 为 `ForwardMode.DECODE`。
4. 批次同样通过 ZeroMQ 发送至 `TpModelWorker`。

### 第六步：TpModelWorker 与 Scheduler 处理 extend 结果
1. `TpModelWorker.forward` 检测 `forward_mode.is_decode()`：
   - 调用 `ForwardBatch.init_from_extend(model_worker_batch, self.model_runner)`。
   - `ForwardBatch.past_key_values` 指向之前缓存的 KV。
2. `ModelRunner.forward` 仅处理增量部分：
   - 读取 `past_key_values`，计算当前 token logits。
   - 处理正则约束、top-p 等策略。
3. `ModelRunner.sample` 计算 `next_token_ids`。若需要返回 logprob，则在同一阶段完成。
4. 结果封装为 `GenerationBatchResult`，主要字段：
   - `next_token_ids`：每个请求一个 token。
   - `next_token_logprobs`（若请求）。
   - `next_token_scores`、`beam_search_info`（如启用）。
5. `Scheduler.process_batch_result`：
   - 将新 token 追加到 `Req.output_token_ids`。
   - 更新 `Req.decode_position` 与 `token_to_kv_pool_allocator` 状态。
   - 检查停止条件：命中 `stop_token_ids`、`max_new_tokens`、自定义回调等。
   - 若完成则调用 `finish_req`，否则继续加入 decode 队列，等待下一次 `extend`。

## 范围与前提
- 当前设计聚焦任务拆解的第一阶段，仅处理单批次的扩散式文本生成流程，不引入完整调度或跨批次共享 kv。
- 全部模型结构沿用 `python/sglang/srt/models/bailing_moe.py`，覆盖张量并行、MoE 以及现有高效算子。
- 仅支持文本生成路径；训练、微调、多模态请求暂不考虑。
- 在 dInfer 中只继承 `IterationSmooth`（`IterSmoothWithVicinityCache`）语义，其余缓存管理逻辑不迁移。

## 任务拆解
1. 第一阶段：在 GPU worker 侧实现 `DiffusionWorker`、`DiffusionIteration`，使其可以独立完成一次扩散前向并产生 token。
2. 第二阶段：基于离线数据集构造简化调度循环，模拟 `Scheduler` 与 kv 管理，验证多批次连续生成。
3. 第三阶段：把实现挂入 sglang 真正的调度框架，复用 `ReqToTokenPool` 与 `token_to_kv_pool_allocator`，并结合 CUDA graph。

## 设计概览：DiffusionWorker 集成思路
- `DiffusionWorker.prefill` 与原生 `TpModelWorker.forward` 接口保持一致，负责生成 prompt embeddings 与初始 logits，并缓存到 `DiffusionBatchState`。
- `DiffusionWorker.extend` 在 `ForwardMode.DECODE` 下启动扩散式 block 解码：对指定 block 进行多次迭代，直至 mask 去除或满足提前终止条件。
- `DiffusionIteration` 管理扩散循环：warmup 次数、continuation 权重、阈值衰减、vicinity cache 更新等都由该组件封装，内部调用 `bailing_moe` 的前向和 sglang 的 CUDA graph 机制。
- 为保证后续阶段易于接入，`DiffusionWorker` 会继续使用 `ForwardBatch`、`SamplingBatchInfo`、`Sampler`、`LogitsProcessor` 等 sglang 现有组件。

## 数据结构与状态映射
- **Req**：新增 `diffusion_state`（第二阶段起启用），保存每个请求的 block 进度、迭代计数、缓存句柄。
- **ScheduleBatch**：prefill 阶段完全沿用原字段；extend 阶段利用 `extend_num_tokens` 描述 block 大小，未来如需一次生成多个 token，可在调度层增加自定义字段。
- **ModelWorkerBatch**：prefill 模式只需在 `DiffusionWorker` 内部补充 block 元信息；extend 模式会把 `extend_prefix_lens`、`extend_input_logprob_token_ids` 传递给扩散逻辑。
- **ForwardBatch**：保留 `forward_mode`、`positions`、`seq_lens`、`sampling_info`；`DiffusionWorker` 会在 block 内更新这些张量以驱动迭代。
- **DiffusionBatchState**（新增）：缓存 prompt embeddings、block 切片、kv 句柄、迭代计数、最近一次 logits。
- **DiffusionBlockContext**（新增）：封装单个 block 的局部视图，包括 mask、阈值、迭代计数、CUDA graph 代理张量。

## 核心组件职责
- **DiffusionWorker**
  - 建立在 `TpModelWorker` 框架之上，初始化 `ModelConfig`、`ServerArgs`、`Sampler`、`CudaGraphRunner`。
  - `prefill`：构造 `DiffusionBatchState`，调用 `DiffusionIteration.allocate_prefill_state` 与 `run_prefill_forward`，返回 `GenerationBatchResult`。
  - `extend`：为每个 block 创建 `DiffusionBlockContext`，循环调用 `DiffusionIteration.run_iteration_step` 直至完成，最后通过 `Sampler` 输出 token。
  - 需要保留的 sglang 组件：`ModelConfig`、`ForwardBatch`、`SamplingBatchInfo`、`LogitsProcessor`、`Sampler`、`CudaGraphRunner`、`GenerationBatchResult`。

- **DiffusionIteration**
  - 负责扩散循环逻辑，兼容 CUDA graph：设定固定 `maximum_unroll`，在循环内部根据 mask 判断是否提前退出。
  - 管理 vicinty cache、continuation 权重、阈值，生成新的 embeddings 与 logits。
  - 需要保留的 sglang 组件：`bailing_moe` 模型及其高效算子、`parallel_state`、`LogitsProcessorOutput` 的结构、`CudaGraphRunner.run_with_graph`。

## Python 接口定义与示例

```python
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import nn

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner


@dataclass
class DiffusionBlockMeta:
    req_index: int
    block_id: int
    start: int
    end: int
    is_finished: bool = False


@dataclass
class DiffusionBatchState:
    prompt_embeddings: torch.Tensor
    block_windows: List[DiffusionBlockMeta]
    kv_handles: Optional[List[object]]
    iteration_counters: List[int]
    cached_logits: Optional[torch.Tensor] = None


@dataclass
class DiffusionIterationResult:
    logits: torch.Tensor
    block_finished: bool
    iterations_used: int


class DiffusionIteration:
    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        block_size: int,
        maximum_unroll: int,
        expected_tpf: int,
        warmup_steps: int,
        cont_weight: float,
        cont_weight_init: float,
        cont_weight_growth: float,
        threshold_decay: float,
        graph_runner: Optional[CudaGraphRunner],
        device: torch.device,
    ) -> None:
        """初始化扩散迭代所需的缓冲区和超参数。"""

    def allocate_prefill_state(self, forward_batch: ForwardBatch) -> DiffusionBatchState:
        """根据 prompt 构建 DiffusionBatchState 并缓存初始 embeddings。"""

    def run_prefill_forward(
        self,
        forward_batch: ForwardBatch,
        batch_state: DiffusionBatchState,
    ) -> LogitsProcessorOutput:
        """执行 prefill 前向，返回可供采样的 logits。"""

    def prepare_block(
        self,
        forward_batch: ForwardBatch,
        batch_state: DiffusionBatchState,
        block_meta: DiffusionBlockMeta,
    ) -> None:
        """在进入扩散循环前初始化 block 的 mask、阈值和局部缓存。"""

    def run_iteration_step(
        self,
        forward_batch: ForwardBatch,
        batch_state: DiffusionBatchState,
        block_meta: DiffusionBlockMeta,
        iter_no: int,
    ) -> DiffusionIterationResult:
        """执行一次扩散迭代，更新 logits、mask 和 embeddings。"""

    def finalize_block(
        self,
        forward_batch: ForwardBatch,
        batch_state: DiffusionBatchState,
        block_meta: DiffusionBlockMeta,
    ) -> None:
        """在 block 完成后写回状态，供下一个阶段使用。"""


class DiffusionWorker:
    def __init__(
        self,
        server_args,
        model_config,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        moe_ep_rank: int,
        graph_runner: Optional[CudaGraphRunner],
        enable_cuda_graph_sampling: bool,
    ) -> None:
        """加载模型、采样器和 DiffusionIteration，准备执行 prefill/extend。"""

    def prefill(self, model_worker_batch: ModelWorkerBatch) -> GenerationBatchResult:
        """运行 prefill，返回包含 logits 或采样 token 的 GenerationBatchResult。"""

    def extend(
        self,
        model_worker_batch: ModelWorkerBatch,
        batch_state: DiffusionBatchState,
    ) -> GenerationBatchResult:
        """对一个 block 执行扩散迭代并生成下一批 token。"""

    def _create_forward_batch(self, model_worker_batch: ModelWorkerBatch) -> ForwardBatch:
        """把 ModelWorkerBatch 转换成 ForwardBatch，适应扩散流程。"""

    def _sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """复用 sglang 采样器产出最终 token。"""
```

```python
# 示例：在单卡环境中运行一次扩散式生成
from types import SimpleNamespace

server_args = SimpleNamespace(tp_size=1, ep_size=1, pp_size=1)
model_config = SimpleNamespace(hidden_size=4096, block_size=64)

graph_runner = CudaGraphRunner()
diffusion_iteration = DiffusionIteration(
    model=bailing_moe_model,
    hidden_size=model_config.hidden_size,
    block_size=model_config.block_size,
    maximum_unroll=4,
    expected_tpf=8,
    warmup_steps=2,
    cont_weight=0.3,
    cont_weight_init=0.15,
    cont_weight_growth=0.02,
    threshold_decay=0.02,
    graph_runner=graph_runner,
    device=torch.device("cuda"),
)

worker = DiffusionWorker(
    server_args=server_args,
    model_config=model_config,
    gpu_id=0,
    tp_rank=0,
    pp_rank=0,
    moe_ep_rank=0,
    graph_runner=graph_runner,
    enable_cuda_graph_sampling=True,
)

# 伪代码：根据实际实现准备 ModelWorkerBatch
prefill_batch = build_prefill_model_worker_batch()
prefill_result = worker.prefill(prefill_batch)

batch_state = diffusion_iteration.allocate_prefill_state(
    ForwardBatch.init_new(prefill_batch, worker)
)

extend_batch = build_extend_model_worker_batch()
generation = worker.extend(extend_batch, batch_state)
print("下一批 token:", generation.next_token_ids)
```
