import math
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if math.isclose(temperature, 0.0):
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

def calculate_op_num(x, hidden_size=4096, mlp_hidden_size = 12288, vocab_size = 126464, num_hidden_layers=32, cache_length=0):
    cfg_factor = 1
    qkv_ops = 4*x.shape[0]*hidden_size*hidden_size*x.shape[1]*2
    attn_ops = x.shape[0]*(cache_length)*x.shape[1]*hidden_size*2
    ffn_ops = 3*x.shape[0]*hidden_size*mlp_hidden_size*x.shape[1]*2
    layer_ops = qkv_ops + attn_ops + ffn_ops
    op_num = cfg_factor * (num_hidden_layers*layer_ops + x.shape[0]*hidden_size*vocab_size*x.shape[1]*2)
    return op_num/1e12 

def calculate_op_num(x, hidden_size=4096, mlp_hidden_size = 12288, vocab_size = 126464, num_hidden_layers=32, cache_length=0):
    cfg_factor = 1
    qkv_ops = 4*x.shape[0]*hidden_size*hidden_size*x.shape[1]*2
    attn_ops = x.shape[0]*(cache_length)*x.shape[1]*hidden_size*2
    ffn_ops = 3*x.shape[0]*hidden_size*mlp_hidden_size*x.shape[1]*2
    layer_ops = qkv_ops + attn_ops + ffn_ops
    op_num = cfg_factor * (num_hidden_layers*layer_ops + x.shape[0]*hidden_size*vocab_size*x.shape[1]*2)
    return op_num/1e12 


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

class TokenArray:
    """ A token array to support read, update and expansion.

    We need to access the tokens that have been generated and write new tokens to the array.
    Some algorithms require to expand the token array.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    """
    def __init__(self, prompt, gen_length, mask_id, device):
        self.prompt = prompt
        self.data = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.gen_length = gen_length

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def device(self):
        return self.data.device

    def expand(self, new_len):
        pass

    def get_generated_tokens(self):
        # TODO(zhengda) we need to define the EOS token
        return self.data[self.data != 126081]

    def __getitem__(self, idx):
        return self.data[:, idx]

    def __setitem__(self, idx, vals):
        self.data[:, idx] = vals

class DistAlignedTokenArray:
    """ A token array to support read, update and expansion in the distributed setting.

    In this setting, each process still contains the full copy of the token array.
    The main difference from TokenArray is that this class makes sure that the length of the token array
    is rounded to the world size.

    Parameters
    ----------
    prompt : Torch.Tensor
        The array that contains the input prompt.
    gen_length : int
        The number of tokens to be generated.
    mask_id : int
        the mask id of the masked tokens
    device : Torch.Device
        The device where the token array is placed on.
    rank : int
        The rank of the process
    world_size : int
        The number of processes.
    """
    def __init__(self, prompt, gen_length, mask_id, device, rank, world_size):
        total_length = prompt.shape[1] + gen_length
        if total_length % world_size != 0:
            total_length = (total_length // world_size + 1) * world_size
        self.data = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(device)
        self.data[:, :prompt.shape[1]] = prompt.clone()
        self.orig_gen_length = gen_length
        self.gen_length = total_length - prompt.shape[1]
        self.prompt = prompt

    @property
    def total_length(self):
        return self.prompt.shape[1] + self.gen_length

    @property
    def device(self):
        return self.data.device

    def get_generated_tokens(self):
        # TODO(zhengda) we need to define the EOS token
        return self.data[self.data != 126081]

    def expand(self, new_len):
        pass

    def __getitem__(self, idx):
        return self.data[:, idx]

    def __setitem__(self, idx, vals):
        self.data[:, idx] = vals

class BlockLoc:
    """ The location of the block in the token array.
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

class BlockIterator:
    """ Block iterator

    This performs block-wise iteration on the input token array for diffusion decoding.

    Parameters
    ----------
    x : TokenArray
        The token array that contains decoded tokens and stores the new generated tokens
    block_length : int
        The length of the block
    """
    def __init__(self, x, block_length):
        self.x = x
        self.iter = 0
        self.block_length = block_length

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        current_block_start = self.x.prompt.shape[1] + self.iter * self.block_length
        if current_block_start >= self.x.total_length:
            raise StopIteration
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, self.x.total_length)
        self.iter += 1
        return BlockLoc(current_block_start, current_block_end), self.x[current_block_start:current_block_end]

class BlockIteratorFactory:
    def create(self, x, block_length):
        return BlockIterator(x, block_length)

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

class ThresholdParallelDecoder(ParallelDecoder):
    """ This decoder deocdes tokens in parallel based on a threshold.

    A token with a confidence score that is larger than a threshold is decoded.
    """
    def __init__(self, temperature, threshold, remasking='low_confidence', mask_id=126336):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.
        """
        mask_index = (x[block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, None, self.threshold)
        x[block_start:block_end][transfer_index] = x0[transfer_index]

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

class PrefixKVCache:
    def __init__(self, model):
        self.model = model
        self.past_key_values = []

    def update(self, x, block_start, block_end):
        """
        """
        output = self.model(x, use_cache=True)
        past_key_values = output.past_key_values

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :block_start],)
        self.past_key_values = new_past_key_values
        return output

    def get_key_values(self):
        return self.past_key_values, None

class DualKVCache:
    def __init__(self, model):
        self.model = model
        self.past_key_values = []
        self.replace_position = None

    def update(self, x, block_start, block_end):
        output = self.model(x, use_cache=True)
        self.past_key_values = output.past_key_values
        self.replace_position = torch.zeros_like(x, dtype=torch.bool)
        self.replace_position[:, block_start:block_end] = 1
        return output

    def get_key_values(self):
        return self.past_key_values, self.replace_position

class DistSPKVCache:
    def __init__(self, rank, world_size, model):
        self.model = model
        self.past_key_values = []

    def update(self, x, current_block_start, current_block_end):
        pass

    def get_key_values(self):
        pass

class KVCacheFactory:
    def __init__(self, cache_type):
        self.cache_type = cache_type

    def create(self, model):
        if self.cache_type == 'prefix':
            return PrefixKVCache(model)
        elif self.cache_type == 'dual':
            return DualKVCache(model)
        else:
            raise ValueError(f'invalid cache type: {self.cache_type}')

def gather_sequence_block(partial_data, partial_start, partial_end, block_start, block_end, rank, world_size):
    """ Gather the wanted block data from the partitioned data.

    Each process contains a partition specified by `partial_start` and `partial_end`.
    The wanted block is located between `block_start` and `block_end`.

    We want to gather the data within the block range from the partitioned data.
    """
    if partial_start >= block_end or partial_end <= block_start:
        # there is no overlap, nothing is needed from partial_data
        arr = partial_data[:, 0:0]
    elif block_start >= partial_start and block_end <= partial_end:
        # the needed block is within partial_data.
        arr = partial_data[:, (block_start - partial_start):(block_end - partial_start)]
    elif block_start <= partial_start and block_end >= partial_end:
        # the needed partition is within the block.
        arr = partial_data
    elif partial_start >= block_start and partial_end >= block_end:
        # the needed block is overlapped in the front of partial_data
        arr = partial_data[:, 0:(block_end - partial_start)]
    else:
        # the needed block is overlapped at the end of partial_data
        arr = partial_data[:, (block_start - partial_start):(partial_end - partial_start)]
    arr = arr.contiguous()

    shape_list = [
            torch.zeros(len(arr.shape), dtype=torch.int64, device=partial_data.device) for _ in range(world_size)
    ]
    dist.all_gather(shape_list, torch.tensor(arr.shape, dtype=torch.int64, device=partial_data.device))
    part_list = [
            torch.zeros(*tuple(shape.tolist()), dtype=partial_data.dtype, device=partial_data.device) for shape in shape_list
    ]
    dist.all_gather(part_list, arr)
    return torch.cat(part_list, dim=1)
