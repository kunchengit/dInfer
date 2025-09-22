import json
import os
import time

import torch
from transformers import AutoTokenizer

from dinfer.decoding.generate_uniform import SlidingWindowDiffusionLLM
from dinfer.decoding.utils import BlockIteratorFactory, KVCacheFactory
from dinfer.decoding import ThresholdParallelDecoder


def benchmark_sw(model, tokenizer, prompt, total_len, block_len, threshold,
                 cache, prefix_look, after_look, warmup_steps, num_test_iter=1,
                 have_warmup=True):
    device = model.device
    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    gen_len = total_len - input_ids.shape[1]
    print("prompt len:", input_ids.shape[1], ", total len:", input_ids.shape[1] + gen_len)

    decoder = ThresholdParallelDecoder(0, threshold=threshold)
    cache_type = cache if cache in ("prefix", "dual") else "dual"
    dllm = SlidingWindowDiffusionLLM(
        model,
        decoder,
        BlockIteratorFactory(),
        KVCacheFactory(cache_type),
        prefix_look=prefix_look,
        after_look=after_look,
        warmup_steps=warmup_steps,
    )

    if have_warmup:
        for _ in range(2):
            _ = dllm._generate(input_ids, gen_length=gen_len, block_length=block_len)

    prev_forwards = dllm.num_forwards
    prev_cache_updates = dllm.cache_updates
    start = time.time()
    num_tokens = 0
    for i in range(num_test_iter):
        out = dllm._generate(input_ids, gen_length=gen_len, block_length=block_len)
        num_tokens += int(out.shape[0]) - int(input_ids.shape[1]) if out.dim() == 1 else (len(out) - int(input_ids.shape[1]))
    stop = time.time()
    total_forward = dllm.num_forwards - prev_forwards
    total_cache_updates = dllm.cache_updates - prev_cache_updates
    print(f"Iter: {i}, Forward: {total_forward}, cache updates: {total_cache_updates}, Time: {stop-start}, FPS: {total_forward/(stop-start)}, TPS: {num_tokens/(stop-start)}")
    print(tokenizer.decode(out[input_ids.shape[1]:], skip_special_tokens=False))


@torch.no_grad()
def main(args):
    torch.cuda.set_device(args.gpu)
    device = torch.device(args.gpu)

    from dinfer.model.modeling_llada_origin import LLaDAModelLM
    model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, init_device=f"cuda:{args.gpu}").eval()
    model = model.to(torch.bfloat16)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    if args.input_data is None:
        if args.prompt is None:
            prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
            m = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        else:
            prompt = args.prompt
        benchmark_sw(
            model,
            tokenizer,
            prompt,
            args.total_len,
            args.block_length,
            args.threshold,
            args.cache,
            args.prefix_look,
            args.after_look,
            args.warmup_steps,
            num_test_iter=args.num_test_iter,
            have_warmup=True,
        )
    else:
        with open(args.input_data, "r") as f:
            data = json.load(f)
        for prompt in data:
            benchmark_sw(
                model,
                tokenizer,
                prompt,
                args.total_len,
                args.block_length,
                args.threshold,
                args.cache,
                args.prefix_look,
                args.after_look,
                args.warmup_steps,
                num_test_iter=args.num_test_iter,
                have_warmup=False,
            )


if __name__ == "__main__":
    import argparse
    torch.multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/mnt/dllm/model_hub/LLaDA-1.5/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input_data", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_test_iter", type=int, default=2)
    parser.add_argument("--total_len", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--cache", type=str, default="dual")
    parser.add_argument("--prefix_look", type=int, default=0)
    parser.add_argument("--after_look", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=1)
    parser.add_argument("--tp", action="store_true")
    args = parser.parse_args()
<<<<<<< HEAD
    main(args)
=======
    main(args)


>>>>>>> update new model
