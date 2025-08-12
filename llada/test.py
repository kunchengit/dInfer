import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

from model.modeling_llada_origin import LLaDAModelLM
from decoding.generate_uniform import BlockIterator, ThresholdParallelDecoder, DiffusionLLM


def test_block_iterator():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    it = BlockIterator(prompt, 10, block_length=5, mask_id=17, device='cpu')
    num_iters = 0
    for block_id, (block_loc, block) in enumerate(it):
        num_iters += 1
        assert block_loc.start == block_id * 5 + prompt.shape[1]
        assert block_loc.end == min((block_id + 1) * 5 + prompt.shape[1], prompt.shape[1] + 10)
    assert num_iters == 2

def test_dist_block_iterator():
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    it = BlockIterator(prompt, 10, block_length=5, mask_id=17, device='cpu')
    num_iters = 0
    for block_id, (block_loc, block) in enumerate(it):
        num_iters += 1
        assert block_loc.start == block_id * 5 + prompt.shape[1]
        assert block_loc.end == min((block_id + 1) * 5 + prompt.shape[1], prompt.shape[1] + 10)
    assert num_iters == 2

def test():
    model_path = "/data/myx/llm/vllm/model/LLaDA-1_5"
    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config)
    model = model.to('cuda:0')

    decoder = ThresholdParallelDecoder(0, threshold=0.9)
    dllm = DiffusionLLM(model, decoder)
    prompt = torch.tensor([1, 2, 3, 4, 5, 6, 7]).view(1, 7)
    dllm._generate(prompt, gen_length=128, block_length=32)

test()
test_block_iterator()
test_dist_block_iterator()
