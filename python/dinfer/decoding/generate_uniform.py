import torch
import numpy as np
import logging

from .utils import TokenArray, DistAlignedTokenArray, gather_sequence_block
from .utils import calculate_op_num

logger = logging.getLogger(__name__)

class DiffusionLLM:
    """ Diffusion LLM inference
    """

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
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
        Torch.Tensor: A tensor of shape (1, L') that contains the prompt tokens and the generated tokens.
            EOS and any tokens after EOS have been removed.
        '''

class BlockRunner:
    """ The class decodes all tokens in a block

    Parameters
    ----------
    diff_iteration : DiffusionIteration
        Run forward computation on a block to decode tokens
    early_stop : bool
        Whether or not to have early stop
    maximum_unroll : int
        The max number of iterations to unroll
    expected_tpf : int
        The expected TPF for loop unrolling.
    """
    def __init__(self, diff_iteration, early_stop, maximum_unroll, expected_tpf):
        self.diff_iteration = diff_iteration
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf

    def decode(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode all tokens in a block.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID

        Returns
        -------
        bool : whether we have decoded all tokens in the sequence.
        """
        while (block == decoder.mask_id).sum() > 0:
            unroll_k = max(min((block == decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1)
            for unroll_i in range(unroll_k):
                self.diff_iteration.forward(model, decoder, x, kv_cache, block, block_loc, block_id)

        if self.early_stop and torch.any(x[block_loc.start:block_loc.end] == decoder.eos_id):
            # Find the first location of EOS and set all tokens after the location to EOS.
            # Here we assume that don't perform remasking.
            # TODO(zhengda) here we assume the batch size is 1.
            x[block_loc.end:] = decoder.eos_id
            return True
        return False

class DiffusionIteration:
    """ A diffusion iteration to decode tokens
    """
    def __init__(self):
        self.num_forwards = 0
        self.cache_updates = 0

    def forward(self, model, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        pass

class BaseDiffusionIteration(DiffusionIteration):
    """ A base implementation of diffusion iteration to decode.
    """
    def __init__(self):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        if kv_cache is None:
            logits = model(x.data).logits[:, block_loc.start:block_loc.end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(x[block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            logits = logits[:, :block_length]
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(block, past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
        

        decoder.decode(logits, block_loc.start, block_loc.end, x)
        self.num_forwards += 1
        self.iter_no += 1

class ShiftDiffusionIteration(DiffusionIteration):
    """ A shift implementation of diffusion iteration to decode.
    """
    def __init__(self, use_shift = False):
        super().__init__()
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ Decode tokens in a forward run on a block.

        The forward run decodes tokens in the input array.

        Parameters
        ----------
        model : pytorch model
            The diffusion LLM
        decoder : ParallelDecoder
            The decoder
        x : TokenArray
            The input tokens. The decoded tokens are also stored in this array.
        kv_cache: KVCache
            The KV-cache
        block : torch.Tensor
            The input IDs of the tokens in the current decoding block.
        block_loc : BlockLoc
            The start and the end of the location of the decoding block.
        block_id : int
            The block ID
        """
        block_start, block_end = block_loc.start-1, block_loc.end-1
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_start, block_end):
            output = model(x.data, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            # TODO(dulun): need to improve efficiency
            x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
            decoder.decode(output.logits[:, block_start:block_end], block_start, block_end, x_shifted)
            x.data[:, 1:] = x_shifted.data
            # update KV-cache
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1

        if kv_cache is None:
            logits = model(x.data).logits[:, block_start:block_end]
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[block_start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_end - block_start
            logits = logits[:, :block_length]
        else:
            # cache position is the position between current_block_start and current_block_end
            past_key_values, replace_position = kv_cache.get_key_values(block_start, block_end)
            logits = model(x[block_start:block_end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
        # TODO(dulun): need to improve efficiency
        x_shifted = TokenArray(x.data[:, 1:], 0, decoder.mask_id, decoder.eos_id, model.device)
        decoder.decode(logits, block_start, block_end, x_shifted)
        x.data[:, 1:] = x_shifted.data
        self.num_forwards += 1
        self.iter_no += 1

class BlockWiseDiffusionLLM(DiffusionLLM):
    """ Diffusion LLM inference

    This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.

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
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, use_shift=False):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        if use_shift:
            self.diff_iteration = ShiftDiffusionIteration()
        else:
            self.diff_iteration = BaseDiffusionIteration()
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)
        

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

    @ torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128):
        ''' Generate tokens with diffusion iterations block by block.
        '''
        assert prompt.shape[0] == 1, "We currently only support batch size = 1."
        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        # We need to reset iter_no at the beginning of generating a sequence.
        self.diff_iteration.iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            decode_compl = self.block_decoder.decode(self.model, self.decoder, x, kv_cache, block, block_loc, block_id)
            if decode_compl:
                break
        logger.info(f'The number of diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()

class IterationSmooth(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if self.inputs_embeds is None:
            self.inputs_embeds = self.h2e(x.data)
        # Update KV-cache
        if kv_cache is not None and kv_cache.require_update(self.iter_no, block_loc.start, block_loc.end):
            output = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            # use the generated output to decode.
            decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            # update KV-cache
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, output.logits, iter_cont_weight)
            kv_cache.update(output.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if kv_cache is None:
            logits = model(inputs_embeds=self.inputs_embeds).logits
            decoder.decode(logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, logits, iter_cont_weight)
        elif kv_cache.cache_type == 'prefix':
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            block_length = block_loc.end - block_loc.start
            decoder.decode(logits[:, :block_length], block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:] = self.h2e(x.data[:, block_loc.start:], mask_index, logits, iter_cont_weight)
        else:
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            # cache position is the position between current_block_start and current_block_end
            logits = model(inputs_embeds=self.inputs_embeds[:, block_loc.start:block_loc.end], past_key_values=past_key_values, use_cache=True,
                    replace_position=replace_position).logits
            decoder.decode(logits, block_loc.start, block_loc.end, x, iter_threshold)
            mask_index = (x.data[:, block_loc.start:block_loc.end] == decoder.mask_id)
            self.inputs_embeds[:, block_loc.start:block_loc.end] = self.h2e(x.data[:, block_loc.start:block_loc.end], mask_index, logits, iter_cont_weight)
        self.num_forwards += 1
        self.iter_no += 1

class IterSmoothDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens block by block.

    The decoding algorithm break the generation sequence into blocks.
    It runs diffusion iterations on the first block and decodes all tokens
    in the block before moving to the next block.
    This is a classifical dLLM decoding algorithm.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8,
                cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        self.diff_iteration = IterationSmooth(self.model, cont_weight, cont_weight_init, cont_weight_growth, threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

class VicinityCacheIteration(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, prefix_look, after_look, warmup_steps):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        if self.iter_no < self.warmup_steps:
            out_full = model(x.data)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(x.data, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        window_input = x.data[:, left_start:right_end]
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(window_input, past_key_values=past_key_values, use_cache=True, replace_position=replace_position)
        self.num_forwards += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x)
        self.iter_no += 1

class VicinityCacheDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens with Vicinity Cache Update.

    The decoding algorithm defines a window to update KV-cache in each diffusion iteration.
    The window can be larger than the decoding block.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = VicinityCacheIteration(prefix_look, after_look, warmup_steps)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates

class IterSmoothWithVicinityCache(DiffusionIteration):
    """ A diffusion iteration to decode tokens
    """
    def __init__(self, model, prefix_look, after_look, warmup_steps,
            cont_weight=0.3, cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        super().__init__()
        self.prefix_look = int(prefix_look)
        self.after_look = int(after_look)
        self.warmup_steps = int(warmup_steps)

        self.cont_weight = cont_weight
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            self.h2e = model.module.h2e
        else:
            self.h2e = model.h2e
        self.cont_weight_init = cont_weight_init
        self.cont_weight_growth = cont_weight_growth
        self.threshold_decay = threshold_decay
        self.inputs_embeds = None
        self.iter_no = 0

    def forward(self, model, decoder, x, kv_cache, block, block_loc, block_id):
        """ The forward computation to decode tokens.
        """
        total_len = x.total_length
        block_start, block_end = block_loc.start, block_loc.end
        left_start = max(0, block_start - self.prefix_look)
        right_end = min(total_len, block_end + self.after_look)

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        if self.inputs_embeds is None:
            self.inputs_embeds = self.h2e(x.data)
        if self.iter_no < self.warmup_steps:
            out_full = model(inputs_embeds=self.inputs_embeds)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            self.iter_no += 1
            return

        if kv_cache.past_key_values is None or (kv_cache.require_update(self.iter_no, block_start, block_end) and block_id > 0):
            out_full = model(inputs_embeds=self.inputs_embeds, use_cache=True)
            self.num_forwards += 1
            decoder.decode(out_full.logits[:, block_start:block_end], block_start, block_end, x, iter_threshold)
            mask_index = (x.data == decoder.mask_id)
            self.inputs_embeds = self.h2e(x.data, mask_index, out_full.logits, iter_cont_weight)
            kv_cache.update(out_full.past_key_values)
            self.cache_updates += 1
            self.iter_no += 1

        iter_cont_weight = min(self.cont_weight_init+self.cont_weight_growth*self.iter_no, self.cont_weight)
        iter_threshold = max(1-self.iter_no*self.threshold_decay, decoder.threshold)
        past_key_values, replace_position = kv_cache.get_key_values(left_start, right_end)
        out_step = model(
                inputs_embeds=self.inputs_embeds[:, left_start:right_end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position
        )

        self.num_forwards += 1
        self.iter_no += 1
        offset = block_start - left_start
        logits_block = out_step.logits[:, offset:offset + (block_end - block_start)]
        decoder.decode(logits_block, block_start, block_end, x, iter_threshold)
        mask_index = (x.data[:, left_start:right_end] == decoder.mask_id)
        self.inputs_embeds[:, left_start:right_end] = self.h2e(x.data[:, left_start:right_end], mask_index, out_step.logits, iter_cont_weight)

class IterSmoothWithVicinityCacheDiffusionLLM(BlockWiseDiffusionLLM):
    """ This diffusion LLM inference generates tokens with vicinity cache and iteration smoothing.
    """
    def __init__(self, model, decoder, iterator_factory, cache_factory, maximum_unroll=4, expected_tpf=8,
                 prefix_look=0, after_look=0, warmup_steps=0, early_stop=True, cont_weight=0.3,
                 cont_weight_init=0.15, cont_weight_growth=0.02, threshold_decay=0.02):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        assert cache_factory is not None, "This class requires a KV-cache."
        self.diff_iteration = IterSmoothWithVicinityCache(model, prefix_look, after_look, warmup_steps,
                cont_weight=cont_weight, cont_weight_init=cont_weight_init, cont_weight_growth=cont_weight_growth,
                threshold_decay=threshold_decay)
        self.block_decoder = BlockRunner(self.diff_iteration, early_stop, maximum_unroll, expected_tpf)

    @property
    def num_forwards(self):
        return self.diff_iteration.num_forwards

    @property
    def cache_updates(self):
        return self.diff_iteration.cache_updates


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
    def generate(self, prompt, gen_length=128, block_length=128):
        '''
        Args:
            prompt: A tensor of shape (1, L).
            gen_length: Generated answer length.
            block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        '''
        op_num = 0
        x = DistAlignedTokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device, self.rank, self.world_size)
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
