

import torch
import numpy as np
import torch.nn.functional as F
import os
import torch.distributed as dist
import time
import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from vllm.config import CompilationConfig, ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config
from vllm.forward_context import set_forward_context
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dinfer.model import FusedOlmoeForCausalLM, LLaDAModelLM
from dinfer.decoding.utils import BlockIteratorFactory, KVCacheFactory
from dinfer.decoding import ThresholdParallelDecoder, BlockWiseDiffusionLLM

def benchmark_gen(rank, model, tokenizer, prompt, gen_len, block_len, threshold, cache, num_test_iter=1, have_warmup=True):
    device = model.device
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    print('prompt len:', input_ids.shape[1], ', total len:', input_ids.shape[1] + gen_len)
    prompt_shape = input_ids.shape

    decoder = ThresholdParallelDecoder(0, threshold=threshold, mask_id=156895, eos_id=156892)
    if cache == 'prefix':
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=KVCacheFactory('prefix'), early_stop=True)
    elif cache == 'dual':
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=KVCacheFactory('dual'), early_stop=True)
    else:
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), early_stop=True)

    # warm up
    if have_warmup:
        for i in range(2):
            out = dllm._generate(input_ids, gen_length=gen_len, block_length=block_len)

    dist.barrier()
    prev_forwards = dllm.num_forwards
    prev_cache_updates = dllm.cache_updates
    start = time.time()
    num_tokens = 0
    for i in tqdm.trange(num_test_iter):
        out = dllm._generate(input_ids, gen_length=gen_len, block_length=block_len)
        num_tokens += len(out) - input_ids.shape[1]
    stop = time.time()
    dist.barrier()
    total_forward = dllm.num_forwards - prev_forwards
    total_cache_updates = dllm.cache_updates - prev_cache_updates
    tps = num_tokens/(stop-start)
    if rank==0:
        print(f'Iter: {i}, Forward: {total_forward}, cache updates: {total_cache_updates}, Time: {stop-start}, FPS: {total_forward/(stop-start)}, TPS: {num_tokens/(stop-start)}')
        print('generated results:', tokenizer.decode(out[input_ids.shape[1]:], skip_special_tokens=False))
    return tps

@ torch.no_grad()
def main(world_size, rank, gpu_id, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    from vllm import distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
    distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    distributed.initialize_model_parallel(args.tp_size, backend='nccl')
    print("[Loading model]")
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        vllm_config = get_current_vllm_config()
        print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)
       
        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        model = FusedOlmoeForCausalLM(config=model_config).eval()
        model.load_weights(args.model_name, torch_dtype=torch.bfloat16)
        if args.tp_size>1 and args.use_tp:
            print('enabling tp')
            model.tensor_parallel(args.tp_size)
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        model = model.to(device)
        prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours?"

        benchmark_gen(rank, model, tokenizer, prompt, args.gen_len, args.block_length, args.threshold, args.cache)
        
        # dist.destroy_process_group()

    
from multiprocessing import Process
import argparse

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/mnt/dllm/fengling/moe/workdir/7bA1b_anneal_15t_0827_500B_further_8k_enneal_train_4k_ep3_v7_1e-5/step45567_converted_hf_fusemoe')
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gen_len', type=int, default=256)
    parser.add_argument('--prompt_length', type=int, default=0)
    parser.add_argument('--block_length', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--low_threshold', type=float, default=0.3)
    parser.add_argument('--parallel_decoding', type=str, default='hierarchy_faster')
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--use_tp', action='store_true')
    args = parser.parse_args()
    procs = []
    print(args)

    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    args.tp_size = len(gpus)
    
    if len(gpus) == 1:
        main(1, 0, gpus[0], args)
    else:
        for i, gpu in enumerate(gpus):
            p = Process(target=main, args=(len(gpus), i, gpu, args))
            p.daemon = True
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
