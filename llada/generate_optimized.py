# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM
import torch.distributed as dist
from generate import generate_with_cache, generate_dist, generate_block_cache



def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12345'
    print(f'rank={rank}, world size={world_size}')
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(world_size, rank, gpu_id):
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)
    setup_distributed(rank, world_size)
    
    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_cache(model, input_ids, world_size=world_size, rank=rank, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    if rank==0:
        print(tokenizer.batch_decode(out[0], skip_special_tokens=False)[0])

if __name__ == '__main__':
    
    gpus = [i for i in range(4)]
    
    if len(gpus) == 1:
        main(1, 0, gpus[0])
    else:
        from multiprocessing import Process
        torch.multiprocessing.set_start_method('spawn')
        procs = []
        for i, gpu in enumerate(gpus):
            p = Process(target=main, args=(len(gpus), i, gpu))
            p.daemon = True
            procs.append(p)
            p.start()
        for p in procs:
            p.join()    
    
