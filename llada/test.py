import logging

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from model.modeling_llada_origin import LLaDAModelLM
from decoding.generate_uniform import DiffusionLLM, DiffusionLLMWithCache
from decoding.generate_fastdllm import generate, generate_with_prefix_cache, generate_with_dual_cache
from decoding.utils import TokenArray, DistAlignedTokenArray, BlockIterator, BlockIteratorFactory, KVCacheFactory
from decoding.utils import ThresholdParallelDecoder


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
    model_path = "/data/myx/llm/vllm/model/LLaDA-1_5"
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config)
    model = model.to('cuda:0')
    decoder = ThresholdParallelDecoder(0, threshold=0.9)

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

logging.basicConfig(level=logging.INFO)
test_diffusion_cached()
test_diffusion_basic()
test_token_array()
test_block_iterator()
