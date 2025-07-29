#!/usr/bin/python
#****************************************************************#
# ScriptName: generate_dist.py
# Author: dulun.dl
# Create Date: 2025-07-29 13:14
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2025-07-29 13:14
# Function: 
#***************************************************************#

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

def calculate_op_num(x, hidden_size=4096, mlp_hidden_size = 12288, vocab_size = 126464, num_hidden_layers=32, cache_length=0):
    cfg_factor = 1
    qkv_ops = 4*x.shape[0]*hidden_size*hidden_size*x.shape[1]*2
    attn_ops = x.shape[0]*(cache_length)*x.shape[1]*hidden_size*2
    ffn_ops = 3*x.shape[0]*hidden_size*mlp_hidden_size*x.shape[1]*2
    layer_ops = qkv_ops + attn_ops + ffn_ops
    op_num = cfg_factor * (num_hidden_layers*layer_ops + x.shape[0]*hidden_size*vocab_size*x.shape[1]*2)
    return op_num/1e12 


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe



@ torch.no_grad()
def generate_with_prefix_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
            
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        output = model(x, use_cache=True)
        past_key_values = output.past_key_values

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        
        past_key_values = new_past_key_values
        nfe += 1
        
        i = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0

            logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:], None, factor)
            x[:, current_block_start:][transfer_index] = x0[transfer_index]
            
            i += 1


    return x, nfe


@ torch.no_grad()
def generate_with_dual_cache(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
            remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0  
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length

        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        # cache init and update
        output = model(x, use_cache=True)
        past_key_values = output.past_key_values
        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if factor is None:
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
        else:
            x0, transfer_index = get_transfer_index_dynamic(output.logits, temperature, remasking, mask_index, x, None, factor)
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        i = 1
        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, current_block_start:current_block_end] = 1
        while True:
            if (x[:, current_block_start:current_block_end] == mask_id).sum() == 0:
                break
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            # cache position is the position between current_block_start and current_block_end
            logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True, replace_position=replace_position).logits

            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, 
                                                x[:, current_block_start:current_block_end], None, factor)
            x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            i += 1

    return x, nfe

@ torch.no_grad()
def generate_block_cache(model, prompt, rank=0, world_size=1, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    op_num = 0
    if prompt.shape[1]%world_size !=0:
        new_prompt = torch.full((prompt.shape[0], prompt.shape[1] + world_size - prompt.shape[1]%world_size), 126081, dtype=prompt.dtype, device=prompt.device)
        new_prompt[:, -prompt.shape[1]:] = prompt
        prompt = new_prompt

    total_length = prompt.shape[1] + gen_length
    if total_length % world_size != 0:
        total_length = (total_length // world_size + 1) * world_size
    x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
    if gen_length==steps:
        steps = total_length-prompt.shape[1]
        last_step = None
    else:
        if gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
    gen_length = total_length - prompt.shape[1]
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = (gen_length+block_length-1) // block_length

    past_key_values = None
    p = 0
    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block + 1) * block_length
        current_block_end = min(current_block_end, total_length)
        # print(rank, num_block, current_block_start, current_block_end, total_length)
        if num_block == num_blocks-1 and last_step is not None:
            steps = last_step
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        past_key_values = None
        if world_size==1:
            output = model(x[:, p:current_block_end].clone(), past_key_values=past_key_values, use_cache=True)
            op_num += calculate_op_num(x[:, p:current_block_end])
            past_key_values = output.past_key_values            
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index[:, p:current_block_end], x[:, p:current_block_end], num_transfer_tokens[:, 0] if threshold is None else None, threshold)
            x[:, p:current_block_end][transfer_index] = x0[transfer_index]        
        else:
            part = (current_block_end-p) // world_size
            output = model(x[:, p+rank*part:p+(rank+1)*part].clone(), use_cache=True)
            op_num += calculate_op_num(x[:, p+rank*part:p+(rank+1)*part])
            partial_logits = output.logits
            past_key_values = output.past_key_values            
            B, L, V = partial_logits.shape
            if L*world_size<=20480:
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index[:, p:current_block_end], x[:, p:current_block_end], num_transfer_tokens[:, 0] if threshold is None else None, threshold)
                x[:, p:current_block_end][transfer_index] = x0[transfer_index]                                 

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start].clone(),)

        past_key_values = new_past_key_values
        nfe += 1
        i = 1
        p = current_block_start

        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            mask_index = (x[:, current_block_start:current_block_end] == mask_id)
            mask_index[:, block_length:] = 0
            if world_size==1:
                logits = model(x[:, current_block_start:current_block_end].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start:current_block_end], cache_length=current_block_end)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]
            else:
                part = (current_block_end - current_block_start) // world_size
                partial_logits = model(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part], cache_length=current_block_end)
                B, L, V = partial_logits.shape
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                # print(rank, logits.shape, partial_logits.shape)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                # print(rank, logits.shape, mask_index.shape, x[:, current_block_start:].shape, num_transfer_tokens[:, i].shape)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:current_block_end], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:current_block_end][transfer_index] = x0[transfer_index]                
            i += 1
    return x, nfe


@ torch.no_grad()
def generate_dist(model, prompt, rank=0, world_size=1, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    op_num = 0
    total_length = prompt.shape[1] + gen_length
    if total_length % world_size != 0:
        total_length = (total_length // world_size + 1) * world_size
    x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
    if gen_length==steps:
        steps = total_length-prompt.shape[1]
        last_step = None
    else:
        if gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
    gen_length = total_length - prompt.shape[1]
    x[:, :prompt.shape[1]] = prompt.clone()

    # assert gen_length % block_length == 0
    num_blocks = (gen_length+block_length-1) // block_length
    
    # print(f'gen_length: {gen_length}, num_blocks: {num_blocks}, steps: {steps}, last_step: {last_step}')



    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block + 1) * block_length
        if num_block == num_blocks-1 and last_step is not None:
            steps = last_step
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            mask_index = (x == mask_id)
            mask_index[:, current_block_end:] = 0
            if world_size==1:
                logits = model(x.clone()).logits
                op_num += calculate_op_num(x)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[transfer_index] = x0[transfer_index]
            else:
                part = x.shape[1] // world_size
                partial_logits = model(x[:, rank*part:(rank+1)*part].clone()).logits
                op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])
                B, L, V = partial_logits.shape
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[transfer_index] = x0[transfer_index]                
            i += 1
    return x, nfe

@ torch.no_grad()
def generate_with_cache(model, prompt, rank=0, world_size=1, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    op_num = 0
    if prompt.shape[1]%world_size !=0:
        new_prompt = torch.full((prompt.shape[0], prompt.shape[1] + world_size - prompt.shape[1]%world_size), 126081, dtype=prompt.dtype, device=prompt.device)
        new_prompt[:, -prompt.shape[1]:] = prompt
        prompt = new_prompt
    # print(rank, prompt.shape, world_size, prompt.shape[1]%world_size)
        

    total_length = prompt.shape[1] + gen_length
    if total_length % world_size != 0:
        total_length = (total_length // world_size + 1) * world_size
    x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
    if gen_length==steps:
        steps = total_length-prompt.shape[1]
        last_step = None
    else:
        if gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
    gen_length = total_length - prompt.shape[1]
    x[:, :prompt.shape[1]] = prompt.clone()

    num_blocks = (gen_length+block_length-1) // block_length


    nfe = 0
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block + 1) * block_length
        current_block_end = min(current_block_end, total_length)
        # print(rank, num_block, current_block_start, current_block_end, total_length)
        if num_block == num_blocks-1 and last_step is not None:
            steps = last_step
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        mask_index = (x == mask_id)
        mask_index[:, current_block_end:] = 0
        if world_size==1:
            output = model(x.clone(), use_cache=True)
            op_num += calculate_op_num(x)
            past_key_values = output.past_key_values            
            x0, transfer_index = get_transfer_index(output.logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
            x[transfer_index] = x0[transfer_index]        
        else:
            part = x.shape[1] // world_size
            output = model(x[:, rank*part:(rank+1)*part].clone(), use_cache=True)
            op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])
            partial_logits = output.logits
            past_key_values = output.past_key_values            
            B, L, V = partial_logits.shape
            if L*world_size<=2048:
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, 0] if threshold is None else None, threshold)
                x[transfer_index] = x0[transfer_index]                 
            else:
                assert part > 256
                if rank == world_size-1:
                    x0, transfer_index = get_transfer_index(partial_logits, temperature, remasking, mask_index[:, -part:], x[:, -part:], num_transfer_tokens[:, 0] if threshold is None else None, threshold)
                else:
                    x0 = torch.empty_like(x[:, -part:])
                    transfer_index = torch.empty_like(x[:, -part:], dtype=torch.bool)
                dist.broadcast(x0, src=world_size-1)
                dist.broadcast(transfer_index, src=world_size-1)
                x[:, -part:][transfer_index] = x0[transfer_index]                 
            

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)    
        past_key_values = new_past_key_values
        nfe += 1
        i = 1

        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, block_length:] = 0
            
            new_past_key_values = []
            for ii in range(len(past_key_values)):
                new_past_key_values.append(())
                for jj in range(len(past_key_values[ii])):
                    new_past_key_values[ii] += (past_key_values[ii][jj].clone(),)    
            past_key_values = new_past_key_values
            
            if world_size==1:
                logits = model(x[:, current_block_start:].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start:], cache_length=total_length)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:][transfer_index] = x0[transfer_index].clone()
            else:
                part = (total_length - current_block_start) // world_size
                partial_logits = model(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part].clone(), past_key_values=past_key_values, use_cache=True).logits
                op_num += calculate_op_num(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part], cache_length=total_length)
                B, L, V = partial_logits.shape
                logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
                dist.all_gather_into_tensor(logits, partial_logits)
                logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x[:, current_block_start:], num_transfer_tokens[:, i] if threshold is None else None, threshold)
                x[:, current_block_start:][transfer_index] = x0[transfer_index]                
            i += 1
    return x, nfe

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
