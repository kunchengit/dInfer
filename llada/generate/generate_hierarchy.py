from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada_origin import LLaDAModelLM

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
def generate_hierarchy(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             mask_id=126336, decoding="origin", **kwargs):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        mask_id: The toke id of [MASK] is 126336.
        kwargs: additional parameters
    '''
    
    threshold = kwargs.get("threshold", None)

    if decoding == "origin":
        get_transfer_index_cur = get_transfer_index
    elif decoding == "hierachy_fast_v2":
        get_transfer_index_cur = partial (get_transfer_index_hierarchy_fast_v2, low_threshold = kwargs["low_threshold"])
    elif decoding == "hierachy_remasking":
        get_transfer_index_cur = partial (get_transfer_index_hierarchy_remask, low_threshold = kwargs["low_threshold"], remask_threshold = kwargs ["remask_threshold"])
    else:
        raise NotImplementedError (decoding)

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_st = prompt.shape[1] + num_block * block_length
        block_ed = prompt.shape[1] + (num_block + 1) * block_length
        x_block = x [:, block_st: block_ed]
        block_mask_index = (x_block == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x_block == mask_id)
            logits = model(x).logits
            # mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            logits_block = logits [:, block_st: block_ed]
            x0, transfer_index = get_transfer_index_cur(logits_block, temperature, mask_index, x_block, 
                num_transfer_tokens[:, i] if threshold is None else None, mask_id, threshold)
            x_block[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, block_st: block_ed] == mask_id).sum() == 0:
                break
    return x, nfe



# Parallel decoding only
def get_transfer_index(logits, temperature, mask_index, x, num_transfer_tokens, mask_id, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    # p = F.softmax(logits.to(torch.float64), dim=-1)
    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    
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


@ torch.no_grad()
def get_transfer_index_hierarchy_fast_v2 (logits, temperature, remasking, mask_index, x, num_transfer_tokens,  mask_id, threshold=None,  low_threshold = None):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)
    

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if  num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True
    
    else:
        for i in range (mask_index.shape[0]):

            mask_i = mask_index[i].int()
            conf_i = confidence[i]

            if low_threshold is not None:
                max_value, max_index = torch.max(conf_i, dim=0)
                if max_value < low_threshold:
                    transfer_index [i, max_index] = True
                    continue


            diff = torch.diff(torch.cat([mask_i[:1]*0, mask_i, mask_i[-1:]*0]))
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]


            if len(starts) > 0:
                max_indices = [s + torch.argmax(conf_i[s:e]) for s, e in zip(starts.tolist(), ends.tolist())]
                transfer_index[i, max_indices] = True
            
            if low_threshold is not None:
                transfer_index [i] = torch.logical_and (transfer_index[i], conf_i > low_threshold) 
                
        if threshold is not None:
            transfer_index = torch.logical_or(transfer_index, confidence > threshold)


    return x0, transfer_index

@ torch.no_grad()
def get_transfer_index_hierarchy_remask (logits, temperature, mask_index, x, num_transfer_tokens,  
                                         mask_id, threshold=None,  low_threshold = None, remask_threshold = 0.4):
    if not math.isclose(temperature, 0.0):
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    else:
        logits_with_noise = logits

    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l


    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l


    lower_index = x0_p < remask_threshold
    remask_index = torch.logical_and (lower_index, torch.logical_not(mask_index))
    mask_new = torch.logical_or (lower_index, mask_index)

    
    confidence = torch.where(mask_new, x0_p, float('-inf'))
    
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    remask_cnt = remask_index.sum (dim = 1)

    
    if  num_transfer_tokens is not None:
        assert threshold is None
        for j in range(confidence.shape[0]):
            _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
            transfer_index[j, select_index] = True
    
    else:
        for i in range (mask_new.shape[0]):

            mask_i = mask_new[i].int()
            conf_i = confidence[i]


            diff = torch.diff(torch.cat([mask_i[:1]*0, mask_i, mask_i[-1:]*0]))
            starts = (diff == 1).nonzero(as_tuple=True)[0]
            ends = (diff == -1).nonzero(as_tuple=True)[0]


            if len(starts) > 0:
                max_indices = [s + torch.argmax(conf_i[s:e]) for s, e in zip(starts.tolist(), ends.tolist())]
                transfer_index[i, max_indices] = True
            
            if low_threshold is not None:
                transfer_index [i] = torch.logical_and (transfer_index[i], conf_i > low_threshold) 
                
            if threshold is not None:
                transfer_index [i] = torch.logical_or(transfer_index [i], conf_i > threshold)

            gap = int((remask_cnt [i] + 1 - transfer_index [i].sum()).item())
            if gap > 0:
                conf_i [transfer_index [i]] = float('-inf')
                values, indices = torch.topk (conf_i, gap, largest=True, sorted=False)
                transfer_index [i][indices] = True
            
    
    remask_index = torch.logical_and (remask_index, torch.logical_not (transfer_index))
    x0 [remask_index] = mask_id
    transfer_index [remask_index] = True

    return x0, transfer_index