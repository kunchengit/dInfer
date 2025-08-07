import yaml
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from postprocess_code import eval_code
import glob
import csv
from enum import Enum

CURRENT_DIR = Path(__file__).resolve().parent
YAML_DIR = CURRENT_DIR.parent / 'yamls'
TASK_DIR = CURRENT_DIR.parent / 'tasks'
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_EVALUATE_OFFLINE']='1'
#os.environ["CUDA_VISIBLE_DEVICES"]='0'

class ModelName(Enum):
    llada15 = "LLaDA-1.5"
    llada_instruct = "LLaDA-8B-Instruct"
    llada_base = "LLaDA-8B-Base"
    llada_unknown = "LLaDA-unknown-version"

class TaskName(Enum):
    humaneval_llada15 = ("humaneval_llada1.5", ModelName.llada15, 0)
    humaneval_llada_instruct = ("humaneval", ModelName.llada_instruct, 0)
    humaneval_llada_base = ("humaneval", ModelName.llada_base, 0)
    humaneval_llada_unknown = ("humaneval", ModelName.llada_unknown, 0)

    gsm8k_llada15 = ("gsm8k_llada1.5", ModelName.llada15, 4)
    gsm8k_llada_instruct = ("gsm8k", ModelName.llada_instruct, 5)
    gsm8k_llada_base = ("gsm8k", ModelName.llada_base, 5)
    gsm8k_llada_unknown = ("gsm8k", ModelName.llada_unknown, 5)

    minerva_math_llada15 = ("minerva_math", ModelName.llada15, 4)
    minerva_math_llada_instruct = ("minerva_math", ModelName.llada_instruct, 4)
    minerva_math_llada_base = ("minerva_math", ModelName.llada_base, 4)
    minerva_math_llada_unknown = ("minerva_math", ModelName.llada_unknown, 4)

    mbpp_llada15 = ("mbpp", ModelName.llada15, 3)
    mbpp_llada_instruct = ("mbpp", ModelName.llada_instruct, 3)
    mbpp_llada_base = ("mbpp", ModelName.llada_base, 3)
    mbpp_llada_unknown = ("mbpp", ModelName.llada_unknown, 3)

    bbh_llada15 = ("bbh", ModelName.llada15, 3)
    bbh_llada_instruct = ("bbh", ModelName.llada_instruct, 3)
    bbh_llada_base = ("bbh", ModelName.llada_base, 3)
    bbh_llada_unknown = ("bbh", ModelName.llada_unknown, 3)

    def __init__(self, task_id, model, fewshot) -> None:
       self.task_id = task_id
       self.model = model
       self.fewshot = fewshot


def process_results(root_dir: str):
    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), desc='merging results...'):
        # if 'Instruct' in dirpath:
        #     continue
        # print(f"当前目录: {dirpath}")
        afcpt = 0
        afcpt_file = os.path.join(dirpath, 'afcpt.json')
        rank_num = 0
        results_file = os.path.join(dirpath, 'all_results.json')
        all_hists = []
        speed = 0
        results_file = None
        speed_our = 0
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if '_afcpt' in filename:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        afcpt += data['average forward calls per token'] 
                        if 'hist' in data:
                            all_hists.append(data["hist"])
                        if 'tokens per second' in data:
                            speed += data['tokens per second']
                        if 'tokens per second our' in data:
                            speed_our += data['tokens per second our']
                rank_num += 1
            if 'results' in filename:
                results_file = os.path.join(dirpath, filename)
            if 'samples_humaneval' in filename:
                if results_file is not None:
                    with open(results_file, 'r', encoding='utf-8') as rfile:
                        results = json.load(rfile)
                    if "post_process_pass@1" in results["results"]["humaneval"]:
                        continue
                    humaneval_result = eval_code(file_path)
                    results["results"]["humaneval"]["post_process_pass@1"] = humaneval_result
                    with open(results_file, 'w', encoding='utf-8') as rfile:
                        rfile.write(json.dumps(results, ensure_ascii=False, indent=4))
        if afcpt != 0:
            with open(afcpt_file, 'w', encoding='utf-8') as f:
                data = {'average forward calls per token': afcpt / rank_num}
                if len(all_hists) > 0:
                    hist = list(map(sum, zip(*all_hists)))
                    data['hist'] = hist
                    data['hist_distrubution'] = [x / sum(hist) for x in hist]
                if speed > 0:
                    data['tokens per second'] = speed / rank_num
                if speed_our > 0:
                    data['tokens per second our'] = speed_our / rank_num
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


def find_results_json(folder):
    # 查找以results开头的json文件
    files = glob.glob(os.path.join(folder, "results*.json"))
    return files[0] if files else None

def find_afcpt_json(folder):
    files = glob.glob(os.path.join(folder, "afcpt.json"))
    return files[0] if files else None

def extract_from_results_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    try:
        if 'gsm8k' in data["results"]:
            val = data["results"]["gsm8k"]["exact_match,flexible-extract"]
        elif 'humaneval' in data["results"]:
            val = data["results"]["humaneval"]["post_process_pass@1"] if "post_process_pass@1" in  data["results"]["humaneval"] else  data["results"]["humaneval"]["pass@1,create_test"] 
        elif 'minerva_math' in data["results"]:
            val = data["results"]["minerva_math"]["math_verify,none"]
        elif 'mbpp' in data["results"]:
            val = data["results"]["mbpp"]["pass_at_1,none"]
        elif 'bbh' in data["results"]:
            val = data["results"]["mbpp"]["exact_match,get-answer"]
        else:
            val = None
    except Exception:
        val = None
    return val

def extract_from_afcpt_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    avg_calls = data.get("average forward calls per token")
    tps = data.get("tokens per second")
    tps_our = data.get("tokens per second our")
    return avg_calls, tps, tps_our

def get_default_fewshot_num(task):
  fewshot_dict = {
  	"humaneval": 0,
    "gsm8k": 5,
    "minerva_math": 4,
    "mbpp": 3,
    "bbh": 3
  }
  return fewshot_dict.get(task, None)

def main():

  os.environ["HF_ALLOW_CODE_EVAL"] = "1"
  os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "True"
  os.environ["HF_DATASETS_OFFLINE"] = "1"
  os.environ["HF_EVALUATE_OFFLINE"] = "1"

  parser = argparse.ArgumentParser()

  parser.add_argument('-y', '--yaml', dest = 'yaml', type=str, required = True)
  args = parser.parse_args()

  yaml_path = YAML_DIR / args.yaml
  with yaml_path.open("r", encoding='utf-8') as f:
    cfgs = yaml.safe_load(f)
  
  
  for cfg in cfgs:
    task = cfg.get('task', None)
    decoding = cfg.get('decoding', None)
    model_path = cfg.get('model_path', None)
    if not all ([task, decoding, model_path]):
      raise TypeError(r"Missing required arguments: 'task', 'decoding', or 'model_path'")
    
    if  ModelName.llada15.value in model_path:
      model = ModelName.llada15
    elif ModelName.llada_instruct in model_path:
      model = ModelName.llada_instruct
    elif ModelName.llada_base in model_path:
      model = ModelName.llada_base
    else:
      model = ModelName.llada_unknown

    task_name = TaskName[f"{task}_{model.name}"]


    length = cfg.get('length', 256)
    block_length = cfg.get('block_length', 32)
    steps = cfg.get ('steps', 128)
    output_dir = cfg.get('output_dir', '/mnt/dllm/dulun.dl/dllm/evaluation_res/')
    num_fewshot = cfg.get('num_fewshot', task_name.fewshot)
    summary_output = cfg.get('summary_output', 'summary.csv')
    show_speed = cfg.get('show_speed', False)
    log_generated_items = cfg.get('log_generated_items', False)
    
    

    

    now = datetime.now()
    ts = now.strftime('%Y-%m-%d_%H:%M:%S')
    output_path = Path(output_dir) / task / model.value / f'genlen{length}' / f'blk{block_length}' / decoding / ts

    ignore_keys = {'task', 'decoding', 'model_path', 'length', 'block_length', 'steps', 'output_dir', 'num_fewshot', 'show_speed', 'log_generated_items', 'summary_output'}
    additional_params = ",".join([f"{k}={v}" for k, v in cfg.items() if k not in ignore_keys])
    
    if task_name == TaskName.humaneval_llada15:
        ext_cmd = f"""accelerate launch eval_llada.py --tasks {task_name.task_id} --apply_chat_template \\
        --confirm_run_unsafe_code --model llada_dist \\
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},show_speed={show_speed},log_generated_items={log_generated_items},save_dir={output_path},{additional_params} \\
        --output_path {output_path} --log_samples \
        ----include_path {TASK_DIR}"""
    elif task_name == TaskName.gsm8k_llada15:
        ext_cmd = f"""accelerate launch eval_llada.py --tasks {task_name.task_id} --num_fewshot {num_fewshot if num_fewshot < 5 else 4} --fewshot_as_multiturn --apply_chat_template \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},show_speed={show_speed},log_generated_items={log_generated_items},save_dir={output_path},{additional_params} \
        --output_path {output_path} \
        --include_path {TASK_DIR}"""
    elif task == "humaneval":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} \\
        --confirm_run_unsafe_code --model llada_dist \\
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},show_speed={show_speed},log_generated_items={log_generated_items},save_dir={output_path},{additional_params} \\
        --output_path {output_path} --log_samples"""
    elif task == "gsm8k":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} --num_fewshot {num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},show_speed={show_speed},log_generated_items={log_generated_items},save_dir={output_path},{additional_params} \
        --output_path {output_path} \
        --include_path {TASK_DIR}"""
    elif task == "minerva_math":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} --num_fewshot {num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},show_speed={show_speed},log_generated_items={log_generated_items},save_dir={output_path},{additional_params} \
        --output_path {output_path}"""
    elif task == "mbpp":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} --num_fewshot {num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},show_speed={show_speed},log_generated_items={log_generated_items},save_dir={output_path},{additional_params} \
        --output_path {output_path}"""
    elif task == "bbh":
      ext_cmd = f"""accelerate launch eval_llada.py --tasks {task} --num_fewshot {num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path={model_path},gen_length={length},steps={steps},block_length={block_length},decoding={decoding},show_speed={show_speed},log_generated_items={log_generated_items},save_dir={output_path},{additional_params} \
        --output_path {output_path}"""
    else:
      #raise TypeError(r"Unsupported task: 'task'")
      continue
    
    print (ext_cmd)

    subprocess.run(ext_cmd, shell = True, check = True)
    # task_list.append(task)
    # path_list.append(output_path)

    process_results(output_path)

    rows = []
    instruct_dir = os.path.join(output_path, model_path.replace("/", "__"))
    results_json = find_results_json(instruct_dir)
    afcpt_json = find_afcpt_json(output_path)

    exact_match_flex = extract_from_results_json(results_json) if results_json else None
    avg_calls, tps, tps_our = (extract_from_afcpt_json(afcpt_json) if afcpt_json else (None, None, None))

    rows.append({
        "task": task,
        "length": length,
        "block_length": block_length,
        "steps": steps,
        "model": model.value,
        "decoding": decoding,
        "num fewshot": num_fewshot, 
        "additional_params": additional_params,
        "score": exact_match_flex,
        "average forward calls per token": avg_calls,
        "tokens per second": tps,
        "tokens per second our": tps_our,
        "eval timestamp": ts
    })

    write_header = not os.path.exists(summary_output) or os.path.getsize(summary_output) == 0
		
    os.makedirs(os.path.dirname(summary_output), exist_ok=True)
    with open(summary_output, "a", newline="") as csvfile:
        fieldnames = ["task","length","block_length","steps","eval timestamp","model","decoding","num fewshot",
                      "additional_params","score","average forward calls per token","tokens per second",
                      "tokens per second our"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
  

if __name__ == "__main__":
  main()