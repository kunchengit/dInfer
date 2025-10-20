# SGLang Illustration Notes

## Q&A Recap

### 1. 离线推理流程讲解
- **Question:** 回归 SGLang 代码，梳理文本生成请求在离线推理引擎中的两次推理流程（从 ScheduleBatch 开始）。  
- **Answer:** 说明了请求如何封装为 `Req`，`ScheduleBatch` 在 `prepare_for_extend()` 中合并张量并分配 KV Cache，随后通过 `ModelWorkerBatch → ForwardBatch` 进入 `ModelRunner.forward()` 完成 prefill；解码阶段 `prepare_for_decode()` 每次仅处理一个 token，复用 KV，并通过 `process_batch_result_prefill/decode` 更新 `Req` 状态和缓存。

### 2. 替换 `TpModelWorker` 的自定义模块
- **Question:** 设计一个最小的自定义模块替换 `TpModelWorker`，保持与现有输入输出兼容、支持 `bailing_moe`。  
- **Answer:** 提议新建 `MinimalTpModelWorker`，保留初始化、`get_worker_info`、`get_memory_pool` 接口；`forward_batch_generation` 中仍使用 `ForwardBatch.init_new()` 和 `ModelRunner.forward()` / `ModelRunner.sample()`；建议直接复用 `ModelRunner` 与底层算子以保持性能。

### 3. 不再保留 `ModelRunner`，只保留前向与采样函数
- **Question:** 如果不想使用 `ModelRunner`，仅实现前向 (`forward`) 与采样 (`sample`) 两个函数，该如何设计以兼容 `bailing_moe`。  
- **Answer:** 提供 `forward_step` 与 `decode_step` 函数雏形：前者直接调用 `BailingMoEForCausalLM.forward`，后者参考 `Sampler.forward()` 处理温度、top-k/p、logprob，并保持与 `ForwardBatch` / `SamplingBatchInfo` 的接口一致。

### 4. 单独测试方案
- **Question:** 如何为上述模块造数据、编写测试以验证与原 `TpModelWorker` 行为一致。  
- **Answer:** 给出 pytest 示例：使用 `ScheduleBatch.init_new()`、`prepare_for_extend()` 构造 `ModelWorkerBatch` 与 `ForwardBatch`，先运行原 `ModelRunner` 获得基准 logits 与 next token，再调用自定义实现比对两者是否完全一致。

