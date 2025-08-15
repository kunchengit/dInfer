import torch
import numpy as np
import torch.nn.functional as F

from .utils import add_gumbel_noise

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

class BlockLoc:
    def __init__(self, start, end):
        self.start = start
        self.end = end

class BlockIterator:
    def __init__(self, prompt, gen_length, block_length, mask_id, device):
        self.x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
        self.x[:, :prompt.shape[1]] = prompt.clone()
        assert gen_length % block_length == 0
        self.num_blocks = gen_length // block_length
        self.iter = 0
        self.prompt = prompt
        self.gen_length = gen_length
        self.block_length = block_length

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        total_length = self.gen_length + self.prompt.shape[1]
        current_block_start = self.prompt.shape[1] + self.iter * self.block_length
        if current_block_start >= total_length:
            raise StopIteration
        current_block_end = current_block_start + self.block_length
        current_block_end = min(current_block_end, total_length)
        self.iter += 1
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]

class DistBlockIterator:
    def __init__(self, prompt, gen_length, block_length, mask_id, device, rank, world_size):
        total_length = prompt.shape[1] + gen_length
        if total_length % world_size != 0:
            total_length = (total_length // world_size + 1) * world_size
        self.total_length = total_length
        self.x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(device)
        self.x[:, :prompt.shape[1]] = prompt.clone()
        self.gen_length = gen_length
        self.block_length = block_length
        gen_length = total_length - prompt.shape[1]
        self.num_blocks = (gen_length + block_length - 1) // block_length
        self.iter = 0

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        current_block_start = self.prompt.shape[1] + self.iter * self.block_length
        if current_block_start >= self.total_length:
            raise StopIteration
        current_block_end = current_block_start + self.block_length
        self.iter += 1
        current_block_end = min(current_block_end, self.total_length)
        return BlockLoc(current_block_start, current_block_end), self.x[:, current_block_start:current_block_end]

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
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, None, self.threshold)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]

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
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]

class DistParallelDecoder(ParallelDecoder):
    """ This decodes tokens in parallel in a distributed fashion based on a threshold.
    """
    def __init__(self, temperature, rank, world_size, remasking='low_confidence', mask_id=126336):
        super().__init__(temperature, remasking, mask_id)
        self.rank = rank
        self.world_size = world_size

    def decode(self, partial_logits, block_start, block_end, x):
        """ Decode the logits in a block.

        """
        curr_x = x[(block_start + partial_logits.start_loc):(block_start + partial_logits.end_loc)]
        mask_index = (curr_x == self.mask_id)
        x0, transfer_index = get_transfer_index(partial_logits, self.temperature, self.remasking, mask_index, curr_x, None, threshold)

        B, L, V = partial_logits.shape
        # TODO(zhengda) why is the decoding here different from the one in the while loop?
        if L * world_size <= 2048:
            # Each process gets all logits in the block and decode themselves.
            logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
            dist.all_gather_into_tensor(logits, partial_logits)
            logits = logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)
            mask_index = (x[:, block_start:block_end] == self.mask_id)
            assert mask_index.shape[1] == logits.shape[1]
            curr_x = x[:, block_start:block_end]
            if self.rank == 0:
                x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, None, self.threshold)
            x[:, block_start:block_end][transfer_index] = x0[transfer_index]
        else:
            # TODO This is a bug. Will be fixed later.
            assert part > 256
            # The last one in the cluster decodes tokens and broadcast them to all other GPUs.
            if rank == world_size - 1:
                # Why does it only decode [-part:]
                x0, transfer_index = get_transfer_index(partial_logits, temperature, remasking, mask_index[:, -part:], x[:, -part:], threshold)
            else:
                x0 = torch.empty_like(x[:, -part:])
                transfer_index = torch.empty_like(x[:, -part:], dtype=torch.bool)
            dist.broadcast(x0, src=world_size-1)
            dist.broadcast(transfer_index, src=world_size-1)
            x[:, -part:][transfer_index] = x0[transfer_index]

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
        output = model(x, use_cache=True)
        self.past_key_values = output.past_key_values
        self.replace_position = torch.zeros_like(x, dtype=torch.bool)
        self.replace_position[:, block_start:block_end] = 1

    def get_key_values(self):
        return self.past_key_values, self.replace_position

class DistSPKVCache:
    def __init__(self, model):
        self.model = model
        self.past_key_values = []

    def update(self, x, current_block_start, current_block_end):
        pass

    def get_key_values(self):
        pass

class DiffusionLLM:
    def __init__(self, model, decoder):
        self.model = model
        self.decoder = decoder

    @ torch.no_grad()
    def _generate(self, prompt, steps=128, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        it = BlockIterator(prompt, gen_length, block_length, self.decoder.mask_id, self.model.device)

        nfe = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            i = 0
            while (block == self.decoder.mask_id).sum() > 0:
                nfe += 1
                logits = self.model(it.x).logits
                # TODO(zhengda) is logits 2-D?
                self.decoder.decode(logits[:, block_loc.start:block_loc.end, :], block_loc.start, block_loc.end, it.x)
                i += 1
        return it.x, nfe

    @ torch.no_grad()
    def generate(self, prompts, steps=128, gen_length=128, block_length=128):
        res = []
        nfes = []
        for prompt in prompts:
            x, nfe = self._generate(prompt, steps, gen_length, block_length)
            res.append(x)
            nfes.append(nfe)
        return res, nfes

class DiffusionLLMWithCache(DiffusionLLM):
    def __init__(self, model, cache_factory, decoder):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder

    @ torch.no_grad()
    def _generate_with(self, prompt, steps=128, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        it = BlockIterator(prompt, gen_length, block_length, self.decoder.mask_id, self.model.device)

        nfe = 0
        kv_cache = self.cache_factory.create(self.model) if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            # Update KV-cache
            if kv_cache is not None and kv_cache.need_update():
                output = kv_cache.update(it.x, block_loc.start, block_loc.end)
                # use the generated output to decode.
                self.decoder.decode(output.logits[:, block_loc.start:block_loc.end, :], block_loc.start, block_loc.end, it.x)
                nfe += 1

            past_key_values, replace_position = kv_cache.get_key_values()
            while (block == mask_id).sum() > 0:

                nfe += 1
                if kv_cache is None:
                    logits = model(it.x).logits[:, block_loc.start:block_loc.end, :]
                elif replace_position is None:
                    logits = model(it.x[:, block_loc.start:], past_key_values=past_key_values, use_cache=True).logits
                    block_length = block_loc.end - block_loc.start
                    logits = logits[:, :block_length, :]
                else:
                    # cache position is the position between current_block_start and current_block_end
                    logits = model(block, past_key_values=past_key_values, use_cache=True,
                                   replace_position=replace_position).logits
                self.decoder.decode(logits, block_loc.start, block_loc.end, it.x)

        return it.x, nfe

def gather_block_logits(partial_logits, partial_start, partial_end, block_start, block_end):
    # TODO(zhengda)
    B, L, V = partial_logits.shape
    logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
    dist.all_gather_into_tensor(logits, partial_logits)
    return logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)


class DiffusionLLMWithSP(DiffusionLLM):
    def __init__(self, model, cache_factory, decoder, rank=0, world_size=1):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.rank = rank
        self.world_size = world_size

    @ torch.no_grad()
    def _generate(model, prompt, steps=128, gen_length=128, block_length=128):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        it = BlockIterator(prompt, gen_length, block_length, self.decoder.mask_id, self.model.device)

        nfe = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == mask_id).sum()>0:
                nfe += 1
                part = x.shape[1] // world_size
                partial_logits = model(x[:, (rank * part):((rank + 1) * part)].clone()).logits
                logits = gather_block_logits(partial_logits, rank * part, (rank + 1) * part, block_loc.start, block_loc.end)
                op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
        return x, nfe

class DiffusionLLMWithSPCache(DiffusionLLM):
    def __init__(self, model, cache_factory, decoder, rank=0, world_size=1):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.rank = rank
        self.world_size = world_size

    @ torch.no_grad()
    def _generate(self, prompt, steps=128, gen_length=128, block_length=128):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        it = BlockIterator(prompt, gen_length, block_length, self.decoder.mask_id, self.model.device)

        nfe = 0
        kv_cache = self.cache_factory.create(self.model)
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            # Update KV-cache
            partial_output = kv_cache.update(x, block_loc.start, block_loc.end)
            # use the generated output to decode.
            logits = gather_block_logits(partial_output.logits, rank * part, (rank + 1) * part, block_loc.start, block_loc.end)
            self.decoder.decode(logits, block_loc.start, block_loc.end, x)
            op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])

            nfe += 1
            while (block == mask_id).sum()>0:
                nfe += 1
                # TODO(zhengda) do we have to duplicate the kv-cache?
                past_key_values = kv_cache.get_past_key_values()
                new_past_key_values = []
                for ii in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for jj in range(len(past_key_values[ii])):
                        new_past_key_values[ii] += (past_key_values[ii][jj].clone(),)
                past_key_values = new_past_key_values

                part = (total_length - block_loc.start) // world_size
                partial_logits = model(x[:, block_loc.start+rank*part:block_loc.start+(rank+1)*part].clone(), past_key_values=past_key_values, use_cache=True).logits
                logits = gather_block_logits(partial_output.logits, block_loc.start + rank * part, block_loc.start + (rank + 1) * part, block_loc.start, block_loc.end)
                op_num += calculate_op_num(x[:, block_loc.start + rank * part:block_loc.start+(rank+1)*part], cache_length=total_length)
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
        return x, nfe
