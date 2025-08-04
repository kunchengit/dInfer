from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada_origin import LLaDAModelLM
import time

from utils import add_gumbel_noise, 

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



@torch.no_grad()
def generate_with_prefixcache_update(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                mask_id=126336, log_flops=False, threshold=None, cache_update_iter=None, eos_early_stop=False, minimal_topk=1, **kwargs):

    '''
    force update cache in some iters.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        mask_id: The toke id of [MASK] is 126336.
    '''
    log_flops = kwargs.get("log_flops", False)
    threshold = kwargs.get("threshold", None)
    cache_update_iter = kwargs.get("cache_update_iter", None)
    eos_early_stop = kwargs.get("eos_early_stop", False)
    minimal_topk = kwargs.get("minimal_topk", 1)



    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    start0 = time.time()

    seq_op_num = 0
    nfe = 0
    cache_update_step=[]
    for num_block in range(num_blocks):
        current_block_start = prompt.shape[1] + num_block * block_length
        current_block_end = prompt.shape[1] + (num_block+1) * block_length
        block_mask_index = (x[:, current_block_start:current_block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        i = 0
        past_key_values = None

        while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
            nfe += 1
            if log_flops:
                if cfg_scale > 0.:
                    cfg_factor = 2
                else:
                    cfg_factor = 1
                actual_shape = x[:, current_block_start:].shape
                op_num = cfg_factor * (32*(4*actual_shape[0]*4096*4096*actual_shape[1]*2 + actual_shape[0]*actual_shape[1]*actual_shape[1]*4096*2+
                        3*actual_shape[0]*4096*12288*actual_shape[1]*2) + actual_shape[0]*4096*126464*actual_shape[1]*2)/ 1e12 
                seq_op_num += op_num        
        
            
            mask_index = (x[:, current_block_start:] == mask_id)
            mask_index[:, current_block_end:] = False

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if cache_update_iter is not None and i%cache_update_iter == 0:
                    # re-calculate kvcache
                    output = model(x_, use_cache=True)
                    logits = output.logits
                    past_key_values = output.past_key_values

                    # update kvcache
                    new_past_key_values = []
                    for k in range(len(past_key_values)):
                        new_past_key_values.append(())
                        for l in range(len(past_key_values[k])):
                            new_past_key_values[k] += (past_key_values[k][l][:, :, :prompt.shape[1] + num_block * block_length],)
                    past_key_values = new_past_key_values

                else:
                    # use kvcache directly
                    logits = model(x_[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if cache_update_iter is not None and i%cache_update_iter == 0:
                    cache_update_step.append(nfe)

                    # re-calculate kvcache
                    output = model(x, use_cache=True)
                    logits = output.logits[:,current_block_start:]
                    past_key_values = output.past_key_values

                    # update kvcache
                    new_past_key_values = []
                    for k in range(len(past_key_values)):
                        new_past_key_values.append(())
                        for l in range(len(past_key_values[k])):
                            new_past_key_values[k] += (past_key_values[k][l][:, :, :prompt.shape[1] + num_block * block_length],)
                    past_key_values = new_past_key_values

                else:
                    logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits

            x0, transfer_index = get_transfer_index_opt(logits, mask_index, x[:,current_block_start:], block_length, num_transfer_tokens[:, i], temperature, remasking,
                                                    threshold=threshold, minimal_topk=minimal_topk)

            x[:, current_block_start:][transfer_index] = x0[transfer_index]

            i+=1
            # early stop: with eos_early_stop tag & + eos token appeared + this block has been unmasked
            if eos_early_stop and (x0[transfer_index] == 126081).any() and not (x[:, current_block_start:current_block_end]==mask_id).any():
                if log_flops:
                    end0 = time.time()
                    print('====sequence flops:', seq_op_num / (end0-start0), 'TFLOPs')
                    print("cache updated as step:", cache_update_step)

                # x[:,current_block_end:] = eos_id
                return x, nfe

    if log_flops:
        end0 = time.time()
        print('====sequence flops:', seq_op_num / (end0-start0), 'TFLOPs')
        print("cache updated as step:", cache_update_step)
    return x, nfe

def get_transfer_index_opt(logits, mask_index, x, block_end, num_transfer_tokens, temperature, remasking, threshold=None, minimal_topk=1):

    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits[mask_index].to(torch.float32), dim=-1).to(logits.dtype)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0[mask_index], -1)), -1)  # b, l
        confidence = torch.full(x0.shape, -np.inf, device=x0.device, dtype=logits.dtype)
        confidence[mask_index] = x0_p
        confidence[:, block_end:] = -np.inf

    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        x0_p[:, block_end:] = -np.inf
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
    else:
        raise NotImplementedError(remasking)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    # print("num_transfer_tokens, topk",num_transfer_tokens[0], minimal_topk)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(minimal_topk, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index






