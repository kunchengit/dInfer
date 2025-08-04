from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada_origin import LLaDAModelLM
import time

from parellel_strategy import get_transfer_index_cache
from utils import get_num_transfer_tokens



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

            x0, transfer_index = get_transfer_index_cache(logits, mask_index, x[:,current_block_start:], block_length, num_transfer_tokens[:, i], temperature, remasking,
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








