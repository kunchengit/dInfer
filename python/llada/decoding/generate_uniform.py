import torch
import numpy as np
import logging

from .utils import TokenArray, DistAlignedTokenArray, gather_sequence_block
from .utils import calculate_op_num

logger = logging.getLogger(__name__)

class DiffusionLLM:
    """ Diffusion LLM inference

    Parameters
    ----------
    model : Torch.Module
        The LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    cache_factory : KVCacheFactory (optional)
        The KV-cache factory that generates a kv-cache for LLM.
    """

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

    @ torch.no_grad()
    def generate(self, prompts, gen_length=128, block_length=128):
        """ Generate tokens with diffusion LLM on a batch of prompts.

        Parameters
        ----------
        prompts : torch.Tensor
            A tensor of shape (b, L) that contains the input prompts.
        gen_length: int
            Generated answer length.
        block_length: int
            Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.

        Returns
        -------
        list[Torch.Tensor]: the generated tokens

        """
        res = []
        for prompt in prompts:
            x = self._generate(prompt, gen_length, block_length)
            res.append(x)
        return res

class BlockWiseDiffusionLLM:
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory=None):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.num_forwards = 0
        self.cache_updates = 0

    @ torch.no_grad()
    def _generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            while (block == self.decoder.mask_id).sum() > 0:
                # Update KV-cache
                if kv_cache is not None and kv_cache.require_update(iter_no, block_loc.start, block_loc.end):
                    output = self.model(x.data, use_cache=True)
                    self.num_forwards += 1
                    # use the generated output to decode.
                    self.decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
                    # update KV-cache
                    kv_cache.update(output.past_key_values)
                    past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
                    self.cache_updates += 1

                if kv_cache is None:
                    logits = self.model(x.data).logits[:, block_loc.start:block_loc.end]
                elif kv_cache.cache_type == 'prefix':
                    logits = self.model(x[block_loc.start:], past_key_values=past_key_values, use_cache=True,
                                        replace_position=replace_position).logits
                    block_length = block_loc.end - block_loc.start
                    logits = logits[:, :block_length]
                else:
                    # cache position is the position between current_block_start and current_block_end
                    logits = self.model(block, past_key_values=past_key_values, use_cache=True,
                                        replace_position=replace_position).logits
                self.num_forwards += 1
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                iter_no += 1

        logger.info(f'The number of diffusion iterations with kv-cache: {self.num_forwards}')
        return x.get_generated_tokens()

class SlidingWindowDiffusionLLM(DiffusionLLM):
    """ This diffusion LLM inference generates tokens in a sliding window manner.

    The decoding algorithm defines a window to decode tokens in each diffusion iteration.
    After each iteration, the decoding window may slide forward to cover more masked tokens.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, update_kv_with_block=False):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.update_kv_with_block = update_kv_with_block
        self.num_forwards = 0
        self.cache_updates = 0

    @ torch.no_grad()
    def _generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        kv_cache = self.cache_factory.create()
        for iter_no, (window_loc, window) in enumerate(it):
            # refresh the entire KV-cache
            if kv_cache.require_update(iter_no, window_loc.start, window_loc.end):
                output = self.model(x.data, use_cache=True)
                self.num_forwards += 1
                # use the generated output to decode.
                self.decoder.decode(output.logits[:, window_loc.start:window_loc.end], window_loc.start, window_loc.end, x)
                # update the kv-cache
                kv_cache.update(output.past_key_values)
                self.cache_updates += 1

            past_key_values, replace_position = kv_cache.get_key_values(window_loc.start, window_loc.end)
            output = self.model(window, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
            # decode in the current window
            self.decoder.decode(output.logits, window_loc.start, window_loc.end, x)
            # update the kv-cache with the data from the current window.
            if self.update_kv_with_block:
                kv_cache.update(output.past_key_values, window_loc.start, window_loc.end)
            self.num_forwards += 1

            # TODO(zhengda) we need to support the expansion of the sequence.

        logger.info(f'The number of diffusion iterations with kv-cache: {self.num_forwards}')
        return x.get_generated_tokens()

class BlockWiseDiffusionLLMWithSP(DiffusionLLM):
    """ Diffusion LLM inference with sequence parallel.

    This class performs diffusion LLM inference with sequence parallel.

    Parameters
    ----------
    rank : int
        The rank of the process
    world_size : int
        The number of processes to perform diffusion LLM inference with sequence parallel.
    model : Torch.Module
        The diffusion LLM model
    decoder : ParallelDecoder
        The decoder that decodes the tokens from the logits computed by the Transformer model
    iterator_facotry : IteratorFactory
        The factory class that generates the iterator on the input token array.
    """
    def __init__(self, rank, world_size, model, decoder, iterator_factory):
        self.model = model
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.rank = rank
        self.world_size = world_size
        self.num_forwards = 0

    @ torch.no_grad()
    def _generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        x = DistAlignedTokenArray(prompt, gen_length, self.decoder.mask_id, self.model.device, self.rank, self.world_size)
        it = self.iterator_factory.create(x, block_length)

        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            while (block == self.decoder.mask_id).sum()>0:
                part = x.total_length // self.world_size
                # TODO(zhengda) How does the model collect KV from other processes.
                partial_logits = self.model(x[(self.rank * part):((self.rank + 1) * part)].clone()).logits
                op_num += calculate_op_num(x[self.rank*part:(self.rank+1)*part])

                logits = gather_sequence_block(partial_logits, self.rank * part, (self.rank + 1) * part, block_loc.start, block_loc.end,
                        self.rank, self.world_size)
                self.decoder.decode(logits, block_loc.start, block_loc.end, x)
                self.num_forwards += 1
        return x.get_generated_tokens()
