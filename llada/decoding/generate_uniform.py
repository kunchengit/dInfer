import torch
import numpy as np
import logging

from .utils import TokenArray

logger = logging.getLogger(__name__)

class DiffusionLLM:
    """ Diffusion LLM inference

    This class performs diffusion LLM inference.

    Parameters
    ----------
    model : Torch.Module
        The diffusion LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    """
    def __init__(self, model, decoder, iterator_factory):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory

    @ torch.no_grad()
    def _generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations.

        Parameters:
        ----------
        prompt: Torch.Tensor
            A tensor of shape (1, L) that contains the input prompt.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        Torch.Tensor: the generated tokens
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        nfe = 0
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == self.decoder.mask_id).sum() > 0:
                logits = self.model(x.data).logits
                # TODO(zhengda) is logits 2-D?
                self.decoder.decode(logits[:, block_loc.start:block_loc.end, :], block_loc.start, block_loc.end, x)
                nfe += 1
        logger.info(f'The number of diffusion iterations: {nfe}')
        return x.get_generated_tokens()

    @ torch.no_grad()
    def generate(self, prompts, gen_length=128, block_length=128):
        res = []
        for prompt in prompts:
            x = self._generate(prompt, gen_length, block_length)
            res.append(x)
        return res

class DiffusionLLMWithCache(DiffusionLLM):
    def __init__(self, model, decoder, iterator_factory, cache_factory):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory

    @ torch.no_grad()
    def _generate_with(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        kv_cache = self.cache_factory.create(self.model) if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            # Update KV-cache
            if kv_cache is not None and kv_cache.need_update():
                output = kv_cache.update(x.data, block_loc.start, block_loc.end)
                # use the generated output to decode.
                self.decoder.decode(output.logits[:, block_loc.start:block_loc.end, :], block_loc.start, block_loc.end, x)

            past_key_values, replace_position = kv_cache.get_key_values()
            while (block == mask_id).sum() > 0:

                if kv_cache is None:
                    logits = model(x.data).logits[:, block_loc.start:block_loc.end, :]
                elif replace_position is None:
                    logits = model(x[block_loc.start:], past_key_values=past_key_values, use_cache=True).logits
                    block_length = block_loc.end - block_loc.start
                    logits = logits[:, :block_length, :]
                else:
                    # cache position is the position between current_block_start and current_block_end
                    logits = model(block, past_key_values=past_key_values, use_cache=True,
                                   replace_position=replace_position).logits
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)

        return x.get_generated_tokens()

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
    def _generate(model, prompt, gen_length=128, block_length=128):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
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
    def _generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            model: Mask predictor.
            prompt: A tensor of shape (1, L).
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
