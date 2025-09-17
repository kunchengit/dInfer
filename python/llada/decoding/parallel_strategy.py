from functools import partial
import math
import torch
import numpy as np
import torch.nn.functional as F

from .utils import add_gumbel_noise

@ torch.no_grad()
def get_transfer_index_hierarchy_fast_v2(logits, temperature, remasking, mask_index, x, num_transfer_tokens,  mask_id, threshold=None,  low_threshold = None):
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
def get_transfer_index_hierarchy_remask(logits, temperature, mask_index, x, num_transfer_tokens,  
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


def get_transfer_index_cache (logits, mask_index, x, block_end, num_transfer_tokens, temperature, remasking, threshold=None, minimal_topk=1):

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

def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
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
        p = F.softmax(logits.to(torch.float64), dim=-1)
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

class ParallelDecoder:
    """ This is a parallel decoder that decodes tokens in a block.
    """
    def __init__(self, temperature, remasking='low_confidence', mask_id=126336):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id

    def block_init(self, block_x, block_id):
        pass

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.

        Parameters
        ----------
        logits : Tensor
            The logits in a block
        block_start : int
            The location of the starting token in the block
        block_end : int
            The location of the ending token in the block.
        x : Tensor
            The tensor where the decoded tokens are written to.
        """

# Parallel decoding only
@ torch.compile(dynamic=True)
def get_transfer_index_threshold(logits, temperature, mask_index, x, num_transfer_tokens, mask_id,
        threshold=None, rm_mask=True, use_float64=False, **kwargs):
    # eos_id=156892
    # mask_id=156895
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if use_float64:
        p = F.softmax(logits.to(torch.float64), dim=-1)
    else:
        p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    
    # gurantee the denoised token will not be the mask_id   
    if rm_mask:
        mask_index = mask_index & (x0 != mask_id)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    if threshold is not None:
        actual_threshold = (torch.max(confidence, dim=1)[0]-1e-5).clamp(-1000, threshold)
        transfer_index = confidence >= actual_threshold
    else:
        topk_values, topk_indices = torch.topk(confidence, k=num_transfer_tokens, dim=1)
        threshold = topk_values[:, -1].clamp(-1000)
        transfer_index = confidence >= threshold
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    return x0, transfer_index

class ThresholdParallelDecoder(ParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold.

    The decoder decodes a token when its confidence score is larger than a threshold.
    """
    def __init__(self, temperature, threshold, remasking='low_confidence', mask_id=126336, early_stop=False, use_float64=False,
            num_mini_transfer_tokens=1):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.early_stop = early_stop
        self.eos_id = 126081
        self.use_float64 = use_float64
        self.num_mini_transfer_tokens = num_mini_transfer_tokens

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.
        """
        mask_index = (x[block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[block_start:block_end]
        x0, transfer_index = get_transfer_index_threshold(logits, self.temperature, mask_index, curr_x,
                self.num_mini_transfer_tokens, self.mask_id, self.threshold, use_float64=self.use_float64)
        if transfer_index.dtype == torch.bool:
            x[block_start:block_end][transfer_index] = x0[transfer_index]
        else:
            x[block_start:block_end][:, transfer_index] = x0[:, transfer_index]
        # If we want to have early stop and there is an EOS decoded in the current block.
        # TODO(zhengda) the code below is not well tested in the unit test.
        if self.early_stop and torch.any(x0 == self.eos_id):
            # Find the first location of EOS and set all tokens after the location to EOS.
            # Here we assume that don't perform remasking.
            # TODO(zhengda) here we assume the batch size is 1.
            idx = int(torch.nonzero(x0[0] == self.eos_id)[0])
            x[(block_start + idx):] = self.eos_id

class FixedParallelDecoder(ParallelDecoder):
    """ This decoder decodes tokens in a fixed number of steps.
    """
    def __init__(self, temperature, steps, remasking='low_confidence', mask_id=126336):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        self.iter = 0

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.
        """
        mask_index = (x[block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[block_start:block_end][transfer_index] = x0[transfer_index]
