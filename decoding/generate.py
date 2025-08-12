class ParallelDecoder:
    """ This is a parallel decoder that decodes tokens in a block.
    """
    def __init__(self, temperature, remasking, mask_id=126336):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id
        self.steps = []

    @property
    def num_blocks(self):
        return len(self.steps)

    def init(self, steps, gen_length, block_length):
        if isinstance(steps, int):
            assert gen_length % block_length == 0
            num_blocks = gen_length // block_length
            assert steps % num_blocks == 0
            self.steps = [steps // num_blocks] * num_blocks
        else:
            self.steps = steps

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
    def __init__(self, temperature, remasking, threshold, mask_id=126336):
        super(self).__init__(temperature, remasking, mask_id)
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
    """ This decoder decodes a fixed number of tokens in a step.
    """
    def __init__(self, temperature, remasking, mask_id=126336):
        super(self).__init__(temperature, remasking, mask_id)
        self.iter = 0

    def block_init(self, block_x, block_id):
        block_mask_index = block_x == mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, self.steps[block_id])
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
    def __init__(self, temperature, remasking, rank, world_size, mask_id=126336):
        super(self).__init__(temperature, remasking, mask_id)
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

    def update(self, x, current_block_start):
        output = self.model(x, use_cache=True)
        past_key_values = output.past_key_values

        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :current_block_start],)
        self.past_key_values = new_past_key_values
        return output

    def get_key_values(self):
        return self.past_key_values, None

class DualKVCache:
    def __init__(self, model):
        self.model = model
        self.past_key_values = []
        self.replace_position = None

    def update(self, x, current_block_start, current_block_end):
        output = model(x, use_cache=True)
        self.past_key_values = output.past_key_values
        self.replace_position = torch.zeros_like(x, dtype=torch.bool)
        self.replace_position[:, current_block_start:current_block_end] = 1

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

    def prepare_x(self, prompt, steps, gen_length, block_length):
        x = torch.full((1, prompt.shape[1] + gen_length), self.decoder.mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        self.decoder.init(steps, gen_length, block_length)
        return x

    @ torch.no_grad()
    def _generate(self, prompt, steps=128, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        x = self.prepare_x(prompt, steps, gen_length, block_length)

        nfe = 0
        for block in range(self.decoder.num_blocks):
            current_block_start = prompt.shape[1] + block * block_length
            current_block_end = current_block_start + block_length
            self.decoder.block_init(x[:, current_block_start:current_block_end], block)
            i = 0
            while (x[:, current_block_start:current_block_end] == self.decoder.mask_id).sum() > 0:
                nfe += 1
                logits = self.model(x).logits
                # TODO(zhengda) is logits 2-D?
                self.decoder.decode(logits[:, current_block_start:current_block_end], current_block_start, current_block_end, x)
                i += 1
        return x, nfe

    @ torch.no_grad()
    def generate(self, prompts, steps=128, gen_length=128, block_length=128):
        res = []
        nfes = []
        for prompt in prompts:
            x, nfe = self._generate(prompt, steps, gen_length, block_length)
            res.append(x)
            nfes.append(nfe)
        return res, nfes

class DiffusionLMWithCache(DiffusionLM):
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
        x = self.prepare_x(prompt, steps, gen_length, block_length)

        nfe = 0
        kv_cache = self.cache_factory.create(self.model) if self.cache_factory is not None else None
        for block in range(self.decoder.num_blocks):
            current_block_start = prompt.shape[1] + block * block_length
            current_block_end = current_block_start + block_length
            self.decoder.block_init(x[:, current_block_start:current_block_end], block)

            # Update KV-cache
            if kv_cache is not None:
                output = kv_cache.update(x, current_block_start, current_block_end)
                # use the generated output to decode.
                self.decoder.decode(output.logits[:, current_block_start:current_block_end], current_block_start, current_block_end, x)
                nfe += 1

            past_key_values, replace_position = kv_cache.get_key_values()
            while (x[:, current_block_start:current_block_end] == mask_id).sum() > 0:
                nfe += 1
                if kv_cache is None:
                    logits = model(x).logits[:, current_block_start:current_block_end]
                elif replace_position is None:
                    logits = model(x[:, current_block_start:], past_key_values=past_key_values, use_cache=True).logits
                    logits = logits[:, :block_length]
                else:
                    # cache position is the position between current_block_start and current_block_end
                    logits = model(x[:, current_block_start:current_block_end], past_key_values=past_key_values, use_cache=True,
                                   replace_position=replace_position).logits
                self.decoder.decode(logits, current_block_start, current_block_end, x)

        return x, nfe

def gather_block_logits(partial_logits, partial_start, partial_end, block_start, block_end):
    # TODO(zhengda)
    B, L, V = partial_logits.shape
    logits = torch.empty(world_size, B, L, V, device=partial_logits.device, dtype=partial_logits.dtype)
    dist.all_gather_into_tensor(logits, partial_logits)
    return logits.permute(1, 0, 2, 3).reshape(B, world_size*L, V)


class DiffusionLLMWithSP(DiffusionLM):
    def __init__(self, model, cache_factory, decoder, rank=0, world_size=1):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.rank = rank
        self.world_size = world_size

    def prepare_x(self, prompt, steps, gen_length, block_length):
        total_length = prompt.shape[1] + gen_length
        if total_length % world_size != 0:
            total_length = (total_length // world_size + 1) * world_size
        x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        if gen_length == steps:
            steps = total_length - prompt.shape[1]
            last_step = None
        elif gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
        gen_length = total_length - prompt.shape[1]
        num_blocks = (gen_length + block_length - 1) // block_length
        block_steps = [steps] * num_blocks
        if last_step is not None:
            block_steps[-1] = last_step
        self.decoder.init(block_steps, gen_length, block_length)
        return x

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
        x = self.prepare_x(prompt, steps, gen_length, block_length)

        nfe = 0
        for block in range(self.decoder.num_blocks):
            current_block_start = prompt.shape[1] + block * block_length
            current_block_end = prompt.shape[1] + (block + 1) * block_length
            self.decoder.block_init(x[:, current_block_start:current_block_end], block)
            while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
                nfe += 1
                part = x.shape[1] // world_size
                partial_logits = model(x[:, (rank * part):((rank + 1) * part)].clone()).logits
                logits = gather_block_logits(partial_logits, rank * part, (rank + 1) * part, current_block_start, current_block_end)
                op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])
                self.decoder.decode(logits, current_block_start, current_block_end, x)
        return x, nfe

class DiffusionLLMWithSPCache(DiffusionLM):
    def __init__(self, model, cache_factory, decoder, rank=0, world_size=1):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.rank = rank
        self.world_size = world_size

    def prepare_x(self, prompt, steps, gen_length, block_length):
        if prompt.shape[1] % world_size !=0:
            # TODO(zhengda) what is 126081?
            new_prompt = torch.full((prompt.shape[0], prompt.shape[1] + world_size - prompt.shape[1]%world_size), 126081, dtype=prompt.dtype, device=prompt.device)
            new_prompt[:, -prompt.shape[1]:] = prompt
            prompt = new_prompt

        total_length = prompt.shape[1] + gen_length
        if total_length % world_size != 0:
            total_length = (total_length // world_size + 1) * world_size
        x = torch.full((prompt.shape[0], total_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        if gen_length==steps:
            steps = total_length-prompt.shape[1]
            last_step = None
        elif gen_length % block_length != 0:
            steps = steps // (gen_length // block_length)
            residual = (total_length - prompt.shape[1]) % block_length
            last_step = max(min(int(residual * steps / block_length)+1, residual), 1)
        else:
            steps = steps // (gen_length // block_length)
            last_step = None
        gen_length = total_length - prompt.shape[1]
        block_steps = [steps] * num_blocks
        if last_step is not None:
            block_steps[-1] = last_step
        self.decoder.init(block_steps, gen_length, block_length)
        return x

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
        x = self.prepare_x(prompt, steps, gen_length, block_length)

        nfe = 0
        kv_cache = self.cache_factory.create(self.model)
        for block in range(self.decoder.num_blocks):
            current_block_start = prompt.shape[1] + block * block_length
            current_block_end = prompt.shape[1] + (block + 1) * block_length
            # TODO(zhengda) how do i get total_length
            #current_block_end = min(current_block_end, total_length)
            self.decoder.block_init(x[:, current_block_start:current_block_end], block)

            # Update KV-cache
            partial_output = kv_cache.update(x, current_block_start, current_block_end)
            # use the generated output to decode.
            logits = gather_block_logits(partial_output.logits, rank * part, (rank + 1) * part, current_block_start, current_block_end)
            self.decoder.decode(logits, current_block_start, current_block_end, x)
            op_num += calculate_op_num(x[:, rank*part:(rank+1)*part])

            nfe += 1
            while (x[:, current_block_start:current_block_end] == mask_id).sum()>0:
                nfe += 1
                mask_index = (x[:, current_block_start:] == mask_id)
                mask_index[:, block_length:] = 0

                # TODO(zhengda) do we have to duplicate the kv-cache?
                past_key_values = kv_cache.get_past_key_values()
                new_past_key_values = []
                for ii in range(len(past_key_values)):
                    new_past_key_values.append(())
                    for jj in range(len(past_key_values[ii])):
                        new_past_key_values[ii] += (past_key_values[ii][jj].clone(),)
                past_key_values = new_past_key_values

                part = (total_length - current_block_start) // world_size
                partial_logits = model(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part].clone(), past_key_values=past_key_values, use_cache=True).logits
                logits = gather_block_logits(partial_output.logits, current_block_start + rank * part, current_block_start + (rank + 1) * part, current_block_start, current_block_end)
                op_num += calculate_op_num(x[:, current_block_start+rank*part:current_block_start+(rank+1)*part], cache_length=total_length)
                self.decoder.decode(logits, current_block_start, current_block_end, x)
    return x, nfe
