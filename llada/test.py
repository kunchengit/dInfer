import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from model.modeling_llada_origin import LLaDAModelLM
from decoding.generate_uniform import DiffusionLLM
from decoding.utils import TokenArray, DistAlignedTokenArray, BlockIterator, BlockIteratorFactory
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

def test():
    model_path = "/data/myx/llm/vllm/model/LLaDA-1_5"
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config)
    model = model.to('cuda:0')

    decoder = ThresholdParallelDecoder(0, threshold=0.9)
    dllm = DiffusionLLM(model, decoder, BlockIteratorFactory())
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    dllm._generate(prompt, gen_length=128, block_length=32)

test_token_array()
test_block_iterator()
test()
