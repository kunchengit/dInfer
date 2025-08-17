import os
import logging
from multiprocessing import Process

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModel, AutoConfig

from model.modeling_llada_origin import LLaDAModelLM
from decoding.generate_uniform import DiffusionLLM, DiffusionLLMWithCache
from decoding.generate_fastdllm import generate, generate_with_prefix_cache, generate_with_dual_cache
from decoding.utils import TokenArray, DistAlignedTokenArray, BlockIterator, BlockIteratorFactory, KVCacheFactory
from decoding.utils import ThresholdParallelDecoder, gather_sequence_block


def test_block_iterator():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    x = TokenArray(prompt, gen_length=10, mask_id=17, device='cpu')
    it = BlockIterator(x, block_length=5)
    num_iters = 0
    for block_id, (block_loc, block) in enumerate(it):
        num_iters += 1
        assert block_loc.start == block_id * 5 + prompt.shape[1]
        assert block_loc.end == min((block_id + 1) * 5 + prompt.shape[1], prompt.shape[1] + 10)
    assert num_iters == 2

def test_token_array():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    arr = TokenArray(prompt, gen_length=20, mask_id=17, device='cpu')
    assert arr.total_length == prompt.shape[1] + 20
    assert torch.all(arr[0:5] == prompt[:, 0:5])
    arr[8:10] = torch.tensor([9, 10]).view(1, 2)

    arr = DistAlignedTokenArray(prompt, gen_length=20, mask_id=17, device='cpu', rank=0, world_size=4)
    assert arr.total_length == prompt.shape[1] + 20 + 1
    assert torch.all(arr[0:5] == prompt[:, 0:5])
    arr[8:10] = torch.tensor([9, 10]).view(1, 2)

def test_diffusion_basic():
    model_path = "/data/myx/llm/vllm/model/LLaDA-1_5"
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config)
    model = model.to('cuda:0')

    decoder = ThresholdParallelDecoder(0, threshold=0.9)
    dllm = DiffusionLLM(model, decoder, BlockIteratorFactory())
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    res = dllm._generate(prompt, gen_length=128, block_length=32)
    res1, nfe = generate(model, prompt, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert len(res) == len(res1)
    assert torch.all(res == res1)

def test_diffusion_cached():
    torch.cuda.set_device(0)
    device = torch.device(0)
    model_path = "/data/myx/llm/vllm/model/LLaDA-1_5"
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    model = model.to(device)
    decoder = ThresholdParallelDecoder(0, threshold=0.9)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she can run 6 kilometers per hour. How many kilometers can she run in 8 hours? "
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    batch_size = 1
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0).repeat(batch_size, 1)

    # Test generation without cache.
    print('Test diffusion LLM without KV-cache')
    dllm = DiffusionLLMWithCache(model, decoder, BlockIteratorFactory())
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    res = dllm._generate(prompt, gen_length=128, block_length=32)
    res1, nfe = generate(model, prompt, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert len(res) == len(res1)
    assert torch.all(res == res1)

    # Test generation with prefix cache
    print('Test diffusion LLM with prefix KV-cache')
    dllm = DiffusionLLMWithCache(model, decoder, BlockIteratorFactory(), KVCacheFactory('prefix'))
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    res = dllm._generate(prompt, gen_length=128, block_length=32)
    res1, nfe = generate_with_prefix_cache(model, prompt, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert len(res) == len(res1)
    assert torch.all(res == res1)

    # Test generation with dual cache
    print('Test diffusion LLM with dual KV-cache')
    dllm = DiffusionLLMWithCache(model, decoder, BlockIteratorFactory(), KVCacheFactory('dual'))
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    res = dllm._generate(prompt, gen_length=128, block_length=32)
    res1, nfe = generate_with_dual_cache(model, prompt, gen_length=128, block_length=32, threshold=0.9)
    res1 = res1[res1 != 126081]
    assert len(res) == len(res1)
    assert torch.all(res == res1)

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def test_worker(rank, world_size, gpu):
    setup_distributed(rank, world_size)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    full_data = torch.arange(100).view(4, 25).to(device)

    # Partition size is smaller than block size.
    block_size = 6
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is larger than block size.
    block_size = 4
    part_size = 6
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    # Partition size is equal to block size.
    block_size = 4
    part_size = 4
    first_part_start = 1
    last_part_end = first_part_start + part_size * world_size
    assert last_part_end <= full_data.shape[1]
    partial_start = first_part_start + part_size * rank
    partial_end = partial_start + part_size
    part_data = full_data[:, partial_start:partial_end]
    # The accessed block must be covered by all parts.
    for block_start in range(first_part_start, last_part_end - block_size):
        block_end = block_start + block_size
        block_data = gather_sequence_block(part_data, partial_start, partial_end, block_start, block_end, rank, world_size)
        assert torch.all(block_data == full_data[:, block_start:block_end])

    dist.destroy_process_group()

def test_dist():
    torch.multiprocessing.set_start_method('spawn')
    num_gpus = 4
    procs = []
    for i, gpu in enumerate(range(num_gpus)):
        p = Process(target=test_worker, args=(i, num_gpus, i))
        procs.append(p)
        p.start()
    for p in procs:
        p.join()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_dist()
    test_diffusion_cached()
    test_diffusion_basic()
    test_token_array()
    test_block_iterator()
