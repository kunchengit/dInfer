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

#@torch.compile()
#@torch.compile(mode='reduce-overhead', fullgraph=False)
def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None,
        optsoftmax=False, force_length=0, force_strength=0.01, eos_id=126081, eot_id=126348,
        prior_front=0, minimal_k=1, use_float64=False):
    # t0 = time.time()
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    # print(logits_with_noise.shape, force_length)
    if force_length>0:
        valid_force_length = min(force_length, logits_with_noise.shape[1])
        delta = torch.arange(force_length, force_length-valid_force_length, -1, dtype=logits.dtype, device=logits.device)*0.01
        logits_with_noise[:, :valid_force_length, eos_id] -=delta
        logits_with_noise[:, :valid_force_length, eot_id] -=delta
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    # t1 = time.time()
    # mask_index shape: b, l
    # print(logits[mask_index])
    if optsoftmax:
        if remasking == 'low_confidence':
            if use_float64:
                p = F.softmax(logits[mask_index].to(torch.float64), dim=-1).to(logits.dtype)
            else:
                p = F.softmax(logits[mask_index], dim=-1).to(logits.dtype)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0[mask_index], -1)), -1) # b, l
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.full(x0.shape, -np.inf, device=x0.device, dtype=logits.dtype)
            # print(confidence.shape, mask_index.shape, x0.shape, x0_p.shape)
            # print(confidence[mask_index].shape, logits.dtype, x0_p.dtype)
            confidence[mask_index] = x0_p
            # x0 = torch.where(mask_index, x0, x)
            # confidence = torch.where(mask_index, x0_p, -np.inf)
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
        else:
            raise NotImplementedError(remasking)
    else:
        if remasking == 'low_confidence':
            if use_float64:
                p = F.softmax(logits.to(torch.float64), dim=-1)
            else:
                p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)
        # print(x.shape, x0.shape, mask_index.shape, logits.shape)
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)
    # t2 = time.time()

    if prior_front!=0:
        confidence += -prior_front+2*prior_front*torch.arange(confidence.shape[1], 0, -1, dtype=confidence.dtype, device=confidence.device)/confidence.shape[1]

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    # t3 = time.time()
    topk = num_transfer_tokens
    topk_values, topk_indices = torch.topk(confidence, k=topk, dim=1)
    transfer_index.zero_()  # 全False

    # 向量化写法，将所有top-k标True
    transfer_index.scatter_(1, topk_indices, True)

    if threshold is not None:
        # top-1保留，其余top-k且<阈值位置要变False
        mask = topk_values[:, minimal_k:] < threshold  # shape (B, 3)
        rows = torch.arange(confidence.size(0), device=mask.device).unsqueeze(1).expand(-1, max(topk-minimal_k, 0))  # shape (B, 3)
        cols = topk_indices[:, minimal_k:]  # shape (B, 3)
        # print(rows.shape, cols.shape, topk, minimal_k)
        # 用mask直接赋值False
        transfer_index[rows[mask], cols[mask]] = False

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
    """ Iterator factory

    This generates iterators for DiffusionLLM to iterate over a sequence.

    Parameters
    ----------
    x : torch.Tensor
        The sequence to iterate over when diffusion LLM generates tokens
    block_length : int
        The block length

    Returns
    -------
    BlockIterator : the block iterator.
    """
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

    The decoder decodes a token when its confidence score is larger than a threshold.
    """
    def __init__(self, temperature, threshold, remasking='low_confidence', mask_id=126336, early_stop=False, use_float64=False):
        super().__init__(temperature, remasking, mask_id)
        self.threshold = threshold
        self.early_stop = early_stop
        self.eos_id = 126081
        self.use_float64 = use_float64

    def decode(self, logits, block_start, block_end, x):
        """ Decode the logits in a block.
        """
        mask_index = (x[block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, None, self.threshold,
                use_float64=self.use_float64)
        x[block_start:block_end][transfer_index] = x0[transfer_index]
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

class KVCache:
    """ KV-cache

    The KV-cache caches the KV of the tokens before and after the block that is being decoded.
    """
    def __init__(self, cache_update_freq=None, cache_type='prefix'):
        self.past_key_values = []
        self.replace_position = None
        self.block_start = None
        self.block_end = None
        self.cache_update_freq = cache_update_freq
        assert cache_type in ['prefix', 'dual']
        self.cache_type = cache_type

    def require_update(self, iter_no, block_start, block_end):
        """ require to update the kv-cache.

        Parameters
        ----------
        iter_no : int
            The diffusion iteration number
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.
        """
        if self.cache_update_freq is None:
            return self.block_start != block_start or self.block_end != block_end
        else:
            return iter_no % self.cache_update_freq == 0 \
                    or (self.block_start != block_start or self.block_end != block_end)

    def update(self, past_key_values, range_start=None, range_end=None):
        """ update the KV-cache

        Parameters
        ----------
        past_key_values : list of list of torch.Tensor
            The key values in all transformer layers.
        range_start : int
            The start of the range that is being updated.
        range_end : int
            The end of the range that is being updated.
        """
        if range_start is None:
            self.past_key_values = past_key_values
        else:
            range_end = range_start + past_key_values[0][0].shape[2] if range_end is None else range_end
            assert range_end - range_start == past_key_values[0][0].shape[2]
            assert len(self.past_key_values) > 0 and self.past_key_values[0][0].shape[2] >= range_end

            # copy the new key-values to the kv-cache.
            for i in range(len(past_key_values)):
                for j in range(len(past_key_values[i])):
                    self.past_key_values[i][j][:, :, range_start:range_end] = past_key_values[i][j]

    def get_key_values(self, block_start, block_end):
        """ Get the key-values given the block that is being decoded.

        Parameters
        ----------
        block_start : int
            The start of the block that is being decoded.
        block_end : int
            The end of the block that is being decoded.

        Returns
        -------
        List[List[torch.Tensor]] : the key-values required to decode the specified block.
        torch.Tensor : the tensor indicates the valid locations in the returned key-values.
        """
        # The key-value cache cannot be empty.
        assert len(self.past_key_values) > 0

        if self.replace_position is not None and self.block_start == block_start and self.block_end == block_end:
            return self.past_key_values, self.replace_position

        # TODO(zhengda) this is a pretty hacky way to find out the batch size and the length of the sequence.
        length = self.past_key_values[0][0].shape[2]
        batch_size = 1
        if self.cache_type == 'prefix':
            self.replace_position = (block_start, length)
        else:
            self.replace_position = (block_start, block_end)
        self.block_start = block_start
        self.block_end = block_end
        return self.past_key_values, self.replace_position

class KVCacheFactory:
    """ KV-cache factory.

    This class generates KV-cache for the diffusion LLM when it runs diffusion iterations.
    """
    def __init__(self, cache_type, cache_update_freq=None):
        self.cache_type = cache_type
        self.cache_update_freq = cache_update_freq

    def create(self):
        return KVCache(cache_update_freq=self.cache_update_freq, cache_type=self.cache_type)

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
