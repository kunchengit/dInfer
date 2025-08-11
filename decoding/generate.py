class ThresholdParallelDecoder:
    """ This decoder deocdes tokens in parallel based on a threshold.
    """
    def __init__(self, temperature, remasking, threshold, mask_id):
        self.threshold = threshold
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id

    def block_init(self, block_x, steps):
        pass

    def decode(self, logits, block_start, block_end, x):
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, None, self.threshold)
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]

class FixedParallelDecoder:
    """ This decoder decodes a fixed number of tokens in a step.
    """
    def __init__(self, temperature, remasking, mask_id):
        self.temperature = temperature
        self.remasking = remasking
        self.mask_id = mask_id
        self.iter = 0

    def block_init(self, block_x, steps):
        block_mask_index = block_x == mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

    def decode(self, logits, block_start, block_end, x):
        mask_index = (x[:, block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[:, block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[:, block_start:block_end][transfer_index] = x0[transfer_index]

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

    def prepare_x(self, prompt, mask_id):
        x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()
        return x

    @ torch.no_grad()
    def _generate(self, prompt, steps=128, gen_length=128, block_length=128,
                  mask_id=126336):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            mask_id: The toke id of [MASK] is 126336.
        '''
        x = self.prepare_x(prompt, mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        nfe = 0
        for num_block in range(num_blocks):
            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length
            self.decoder.block_init(x[:, current_block_start:current_block_end], steps)
            i = 0
            while (x[:, current_block_start:current_block_end] == mask_id).sum() > 0:
                nfe += 1
                logits = self.model(x).logits
                # TODO(zhengda) is logits 2-D?
                self.decoder.decode(logits[:, current_block_start:current_block_end], current_block_start, current_block_end, x)
                i += 1
        return x, nfe

    @ torch.no_grad()
    def generate(self, prompts, steps=128, gen_length=128, block_length=128, mask_id=126336):
        res = []
        nfes = []
        for prompt in prompts:
            x, nfe = self._generate(prompt, steps, gen_length, block_length, mask_id)
            res.append(x)
            nfes.append(nfe)
        return res, nfes

class DiffusionLMWithCache(DiffusionLM):
    def __init__(self, model, cache_factory, decoder):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder

    @ torch.no_grad()
    def _generate_with(self, prompt, steps=128, gen_length=128, block_length=128, mask_id=126336):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            steps: Sampling steps, less than or equal to gen_length.
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
            mask_id: The toke id of [MASK] is 126336.
        '''
        x = self.prepare_x(prompt, mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        nfe = 0
        kv_cache = self.cache_factory.create(self.model) if self.cache_factory is not None else None
        for num_block in range(num_blocks):
            current_block_start = prompt.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length
            self.decoder.block_init(x[:, current_block_start:current_block_end], steps)

            # Update KV-cache
            if kv_cache is not None:
                output = kv_cache.update(x)
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
