import os
import multiprocessing as mp

import torch
import torch.distributed as dist
from vllm import distributed as vllm_dist
from transformers import AutoConfig
from vllm.config import ParallelConfig
from vllm.config import VllmConfig, set_current_vllm_config, get_current_vllm_config

from .parallel_strategy import ThresholdParallelDecoder
from .utils import KVCacheFactory, BlockIteratorFactory
from .generate_uniform import IterSmoothWithVicinityCacheDiffusionLLM, IterSmoothDiffusionLLM, VicinityCacheDiffusionLLM, BlockWiseDiffusionLLM
from ..model.modeling_fused_olmoe import FusedOlmoeForCausalLM
from ..model.modeling_llada import LLaDAModelLM

class SamplingParams:
    """ The parameters used for sampling a sequence.

    Parameters
    ----------
    threshold : float
        The threshold used for threshold-based parallel decoding algorithm.
    cache : str
        The kv-cache type. Valid values include 'prefix', 'dual' and ''.
    temperature : float
        The temperature used for decoding tokens.
    early_stop : bool
        Whether to stop generating tokens after encountering an EOS.
    cont_weight : float
        This is used by IterSmooth algorithm.
    prefix_look : int
        This is used by vicinity KV-cache refresh algorithm.
        This determines the number of tokens before the decoding block that should recompute key and value states in every diffusion iteration.
    after_look : int
        This is used by vicinity KV-cache refresh algorithm.
        This determines the number of tokens after the decoding block that should recompute key and value states in every diffusion iteration.
    warmup_steps : int
        This is used by vicinity KV-cache refresh algorithm.
        This determines the number of steps at the beginning that we need to refresh key and value states of the entire sequence.
    enable_torch_compile : bool
        Whether to use torch compile for the model code.
    mask_id : int
        The mask ID
    eos_id : int
        The EOS ID
    """
    def __init__(self, threshold=0.9, cache='dual', temperature=0., early_stop=True, cont_weight=0.3,
            prefix_look=16, after_look=16, warmup_steps=4, enable_torch_compile=True, mask_id=156895, eos_id=156892):
        self.threshold = threshold
        self.cache = cache
        self.temperature = temperature
        self.early_stop = early_stop
        self.cont_weight = cont_weight
        self.prefix_look = prefix_look
        self.after_look = after_look
        self.warmup_steps = warmup_steps
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.enable_torch_compile = enable_torch_compile

def init_generator(model, sample_params):
    decoder = ThresholdParallelDecoder(temperature=sample_params.temperature, threshold=sample_params.threshold,
            mask_id=sample_params.mask_id, eos_id=sample_params.eos_id)
    if sample_params.cache == 'prefix' or sample_params.cache == 'dual':
        cache_factory = KVCacheFactory(sample_params.cache)
    else:
        cache_factory = None
    if cache_factory is not None and sample_params.cont_weight > 0:
        dllm = IterSmoothWithVicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=cache_factory,
                early_stop=sample_params.early_stop, cont_weight=sample_params.cont_weight, prefix_look=sample_params.prefix_look,
                after_look=sample_params.after_look, warmup_steps=sample_params.warmup_steps)
    elif cache_factory is not None:
        dllm = VicinityCacheDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=cache_factory,
                early_stop=sample_params.early_stop, prefix_look=sample_params.prefix_look,
                after_look=sample_params.after_look, warmup_steps=sample_params.warmup_steps)
    elif sample_params.cont_weight > 0:
        dllm = IterSmoothDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=None,
                early_stop=sample_params.early_stop, cont_weight=sample_params.cont_weight)
    else:
        dllm = BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(), cache_factory=None,
                early_stop=sample_params.early_stop)
    return dllm

def generate(dllm, device, req_q, res_q):
    while True:
        data = req_q.get()
        if isinstance(data, str):
            assert data == 'stop'
            break
        else:
            input_ids, gen_len, block_len = data
        out = dllm.generate(input_ids, gen_length=gen_len, block_length=block_len)
        if res_q is not None:
            res_q.put(out)

def moe_server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(master_port)
    vllm_dist.init_distributed_environment(world_size, rank, 'env://', rank, 'nccl')
    vllm_dist.initialize_model_parallel(world_size, backend='nccl')
    # setup EP
    parallel_config = ParallelConfig(enable_expert_parallel = True)
    with set_current_vllm_config(VllmConfig(parallel_config = parallel_config)):
        vllm_config = get_current_vllm_config()
        print("EP Enabled:", vllm_config.parallel_config.enable_expert_parallel)

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = FusedOlmoeForCausalLM(config=model_config).eval()
        model.load_weights(model_path, torch_dtype=torch.bfloat16)
        if world_size > 1:
            model.tensor_parallel(world_size)
        if sample_params.enable_torch_compile:
            model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
        model = model.to(device)

        dllm = init_generator(model, sample_params)
        generate(dllm, model.device, req_q=q, res_q=res_q)

    dist.destroy_process_group()

def server_process(model_path, sample_params, world_size, rank, gpu_id, q, res_q, master_port):
    torch.cuda.set_device(gpu_id)
    device = torch.device(gpu_id)

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(master_port)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    config = AutoConfig.from_pretrained(model_path)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, config=config).eval()
    if world_size > 1:
        model.tensor_parallel(world_size)
    if sample_params.enable_torch_compile:
        model.forward = torch.compile(model.forward, mode='reduce-overhead', fullgraph=False, dynamic=True)
    model = model.to(device)

    dllm = init_generator(model, sample_params)
    generate(dllm, model.device, req_q=q, res_q=res_q)

    dist.destroy_process_group()

class ServerHandle:
    def __init__(self):
        self.procs = []
        self.req_qs = []
        self.res_q = None

    def add_request(self, req):
        assert len(self.req_qs) != 0 and len(self.req_qs) == len(self.procs)
        for q in self.req_qs:
            q.put(req)

    def get_response(self):
        return self.res_q.get()

    def start_server(self, model_path, is_moe, sample_params, server_port, num_gpus):
        ctx = mp.get_context('spawn')
        assert len(self.procs) == 0, 'The server is already running.'
        procs = []
        req_qs = []
        for i in range(num_gpus):
            if i == 0:
                res_q = ctx.Queue()
                self.res_q = res_q
            else:
                res_q = None
            q = ctx.Queue()
            req_qs.append(q)
            if is_moe:
                p = ctx.Process(target=moe_server_process, args=(model_path, sample_params, num_gpus, i, i, q, res_q, server_port))
            else:
                p = ctx.Process(target=server_process, args=(model_path, sample_params, num_gpus, i, i, q, res_q, server_port))
            p.daemon = True
            procs.append(p)
            p.start()
        self.procs = procs
        self.req_qs = req_qs

    def is_running(self):
        return len(self.procs) != 0

    def stop_running(self):
        for q in self.req_qs:
            q.put('stop')
        for p in self.procs:
            p.join()

        self.procs = []
        self.req_qs = []
        self.req_q = None

handle = ServerHandle()

class DiffusionLLMServing:
    """ Serving dLLM inference.

    This is an experimental feature to enable serving in dInfer.
    This class creates multiple processes to enable dLLM inference in the background. A new request is sent to the background processes
    for model inference and the result is sent back to the main process.

    Parameters
    ----------
    model : str
        The model path
    is_moe : bool
        Whether this is a MOE model. This leads to using different model code and inference code.
    sample_params : SamplingParams
        The parameters used in sampling.
    server_port : int
        The port for communication between the background process.
    num_gpus : int
        The number of GPUs used for parallel computation.
    """
    def __init__(self, model, is_moe=True, sample_params=None, server_port=12345, num_gpus=None):
        if sample_params is None:
            sample_params = SamplingParams()
        if num_gpus is None:
            num_gpus = torch.cuda.device_count()
        if not handle.is_running():
            handle.start_server(model, is_moe, sample_params, server_port, num_gpus)

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
        prompt = prompt.cpu()
        handle.add_request((prompt, gen_length, block_length))
        return handle.get_response()

    def stop_serving(self):
        """ Stop model serving.
        """
        handle.stop_running()
