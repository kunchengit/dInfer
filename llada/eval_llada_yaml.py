import yaml
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

CURRENT_DIR = Path(__file__).resolve().parent
YAML_DIR = CURRENT_DIR.parent / 'yamls'

def main():

  os.environ["HF_ALLOW_CODE_EVAL"] = 1
  os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = True
  os.environ["HF_DATASETS_OFFLINE"] = 1
  os.environ["HF_EVALUATE_OFFLINE"] = 1

  parser = argparse.ArgumentParser()

  parser.add_argument('-y', '--yaml', dest = 'yaml', type=str, required = True)
  args = parser.parse_args()

  yaml_path = YAML_DIR / args.yaml
  with yaml_path.open("r", encoding='utf-8') as f:
    cfgs = yaml.safe_load(f)
  
  path_list = []
  task_list = []
  for cfg in cfgs:
    task = cfg.get('task', None)
    decoding = cfg.get('decoding', None)
    length = cfg.get('length', 256)
    block_length = cfg.get('block_length', 32)
    steps = cfg.get ('steps', 128)
    model_path = cfg.get('model_path', None)
    output_dir = cfg.get('output_dir', '/mnt/dllm/dulun.dl/dllm/evaluation_res/')
    num_fewshot = cfg.get('num_fewshot', 5)

    if not all ([task, decoding, model_path]):
      raise TypeError(r"Missing required arguments: 'task', 'decoding', or 'model_path'")

    if  "LLaDA-1.5" in model_path:
      model = "LLaDA-1.5"
    elif "LLaDA-8B-Instruct" in model_path:
      model = "LLaDA-8B-Instruct"
    elif "LLaDA-8B-Base" in model_path:
      model="LLaDA-8B-Base"
    else:
      model="LLaDA-unknown-version"

    now = datetime.now()
    ts = now.strftime('%Y-%m-%d_%H:%M:%S')
    output_path = Path(output_dir) / task / model / f'genlen{length}' / f'blk{block_length}' / decoding / ts

    ignore_keys = {'task', 'decoding', 'model_path', 'length', 'block_length', 'steps', 'output_dir', 'num_fewshot'}
    additional_params = ",".join([f"{k}={v}" for k, v in cfg.items() if k not in ignore_keys])
    
    if task == "humaneval":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},{additional_params} \
        --output_path {output_path} --log_samples"""
    elif task == "gsm8k"
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} num_fewshot {num_fewshot}\
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},{additional_params} \
        --output_path {output_path} --log_samples"""
    elif task == "minerva_math":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} num_fewshot {num_fewshot}\
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},{additional_params} \
        --output_path {output_path} --log_samples"""
    elif task == "mbpp"
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} num_fewshot {num_fewshot}\
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},{additional_params} \
        --output_path {output_path} --log_samples"""
    elif task == "bbh":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} num_fewshot {num_fewshot}\
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},{additional_params} \
        --output_path {output_path} --log_samples"""
    else:
      #raise TypeError(r"Unsupported task: 'task'")
      continue
    
    subprocess.run([ext_cmd], check = True)
    task_list.append(task)
    path_list.append(output_path)
  




if __name__ == "__main__":
  main()