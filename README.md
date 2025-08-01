# LLaDA Eval
## IDE Experiments
### Preparation
- Image selection: reg.docker.alibaba-inc.com/aii/aistudio:12910142-20250729192318
- NAS DIR: dllm (50T)
  ```bash
    mkdir /mnt/dllm
    {mount_command}
    cp /mnt/dllm/dulun.dl/dllm_decoding/huggingface.zip ~/.cache
    cd ~/.cache
    unzip huggingface.zip
    rm -rf huggingface.zip
  ```
  - mount cmd is confidential, please reach out to dulun.dl for usage
### Execution
- A usage case for yaml mode:
	```bash
		cd {Fastdllm repo path}/llada
		python eval_llada_yaml.py -y fastdll_humaneval.yaml
	```
- A usage case for scripts

## Job Submission
- A usage case for yaml mode:
	```bash
		cd {Fastdllm repo path}/job_submit
		python llada_eval_job.py --yaml fastdll_humaneval.yaml --mount {mount_cmd}
	```
- A usage case for script mode:
	```bash
		cd {Fastdllm repo path}/job_submit
		python llada_eval_job.py --script eval_humaneval.sh --mount {mount_cmd}
	```

# Decoding Parameters
## Fastdllm
- `decoding`: 'fastdllm'
- `remasking`: {'low_confidence' | 'random'} (default: 'low_confidence')
- `use_cache`: boolean (default: False)
- `threshold`: {float (0 - 1) | None} (default: None)
- `factor` : {flaot (0 - 1), None} (default: None)
- `dual_cache`: boolean (default: False)

## Hierarchy Decoding
### hierarchy_fast
- `decoding`: 'hierarchy_fast_v2'
- `threshold`: {float (0 - 1) | None} (default: None)
- `low_threshold`: {float (0 - threshold) | None} (default: None)

### hierarchy_remasking
- `decoding`: 'hierarchy_remasking'
- `threshold`: {float (0 - 1) | None} (default: None)
- `low_threshold`: {float (0 - threshold) | None} (default: None)
- `remask_threshold`: 

## Other Parameters
- `length`: int (default: 256)
- `block_length`: int (default: 32)
- `steps`:  int (default: 128)
- `show_speed`: boolean (default: False)
- `log_generated_items`: boolean (default: False)
- `summary_output`: String (default: 'summary.csv')
- `output_dir`: String
- `model_path`: String