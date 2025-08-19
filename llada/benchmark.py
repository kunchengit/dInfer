

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel


import torch.distributed as dist

import time
import tqdm

from decoding.generate_uniform import BlockWiseDiffusionLLM, SlidingWindowDiffusionLLM, BlockWiseDiffusionLLMWithSP
from decoding.utils import TokenArray, DistAlignedTokenArray, BlockIterator, BlockIteratorFactory, KVCacheFactory
from decoding.utils import ThresholdParallelDecoder, gather_sequence_block, BlockLoc

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
@ torch.no_grad()
def main(world_size, rank, gpu_id, args):
    print('started', world_size, rank, gpu_id, args)
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    if args.tp:
        from model.modeling_llada_origin import LLaDAModelLM        
        model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, init_device='cuda:'+str(gpu_id)).eval()
        import vllm
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '23456'        
        print(vllm.distributed.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl'))
        vllm.distributed.initialize_model_parallel(world_size, backend='nccl')        
        if world_size>1:
            model.tensor_parallel(world_size)
            
        model = model.to(torch.bfloat16)
        model = model.to(device)
        # model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
        model.forward = torch.compile(model.forward, fullgraph=True, dynamic=False)
    else:
        from model.modeling_llada import LLaDAModelLM
        setup_distributed(rank, world_size)
        model = LLaDAModelLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, init_device='cuda:'+str(gpu_id)).eval()
        model = torch.compile(model, mode='reduce-overhead', fullgraph=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = model.to(device)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    batch_size = args.batch_size
    num_iter = args.num_iter
    #num_parallel = args.num_parallel
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0).repeat(batch_size, 1)
    print(input_ids.shape, input_ids.shape[1] + args.gen_len)

    block_length=args.block_length
    gen_len = args.gen_len
    prompt_shape = input_ids.shape

    decoder = ThresholdParallelDecoder(0, threshold=args.threshold)
    if args.cache == 'prefix':
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory('prefix'))
    elif args.cache == 'dual':
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory('dual'))
    else:
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory())

    # warm up
    for i in range(2):
        out = dllm._generate(input_ids, gen_length=gen_len, block_length=block_length)

    total_shape = out.shape
    
    start = time.time()
    for i in tqdm.trange(num_iter):   
        out = dllm._generate(input_ids, gen_length=gen_len, block_length=block_length)
    stop = time.time()
    total_forward = dllm.num_forwards
    if rank==0:
        print(f'Forward: {total_forward}, Time: {stop-start}, FPS: {total_forward/(stop-start)}, TPS: {batch_size*gen_len*num_iter/(stop-start)}')
        #for i in range(1):
        #    print(tokenizer.decode(out[i, input_ids.shape[1]:], skip_special_tokens=False))
    dist.destroy_process_group()
    
from multiprocessing import Process
import argparse

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/data/myx/llm/vllm/model/LLaDA-1_5')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_iter', type=int, default=4)
    parser.add_argument('--num_parallel', type=int, default=4)
    parser.add_argument('--gen_len', type=int, default=256)
    parser.add_argument('--block_length', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--cache', type=str, default='')
    parser.add_argument('--block_diffusion', action='store_true')
    parser.add_argument('--tp', action='store_true')
    args = parser.parse_args()
    procs = []
    print(args)

    gpus = [int(gpu) for gpu in args.gpu.split(',')]
    
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
