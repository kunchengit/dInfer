import functools
import os
import json
from tqdm import tqdm
from postprocess_code import eval_code
import glob
import csv
import argparse
os.environ['HF_DATASETS_OFFLINE']='1'
os.environ['HF_EVALUATE_OFFLINE']='1'
def count_forward_calls(forward):
    """A decorator for counting the number of calls to the model's forward method."""
    @functools.wraps(forward)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_forward_call_count'):
            self._forward_call_count = 0
        self._forward_call_count += 1
        return forward(self, *args, **kwargs)
    return wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description='tools to process and merge results')
    parser.add_argument('--type', help='The type of operation', type=str)
    parser.add_argument('--res_dir', help='The dir of the inference output', type=str)
    args = parser.parse_args()
    return args

def process_results(root_dir: str):
    for dirpath, dirnames, filenames in tqdm(os.walk(root_dir), desc='merging results...'):
        # if 'Instruct' in dirpath:
        #     continue
        print(f"当前目录: {dirpath}")
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
                humaneval_result = eval_code(file_path)
                if results_file is not None:
                    with open(results_file, 'r', encoding='utf-8') as rfile:
                        results = json.load(rfile)
                    results["results"]["humaneval"]["post_process_pass@1"] = humaneval_result
                    with open(results_file, 'w', encoding='utf-8') as rfile:
                        rfile.write(json.dumps(results, ensure_ascii=False))
            # else:
            #     with open(file_path, 'r', encoding='utf-8') as in_file, open(results_file, 'a', encoding='utf-8') as out_file:
            #         for line in in_file:
            #             out_file.write(line)
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


def parse_dir_name(dir_path):
    # 解析目录名
    # .../hierarchy_decoding/type/task/length_blocklength_threshold_lowthreshold_remaskthreshold
    parts = dir_path.strip(os.sep).split(os.sep)
    type_, task, params = parts[-3], parts[-2], parts[-1]
    length, block_length, threshold, low_threshold, remask_threshold = params.split("_")
    return type_, task, length, block_length, threshold, low_threshold, remask_threshold

def find_results_json(folder):
    # 查找以results开头的json文件
    files = glob.glob(os.path.join(folder, "results*.json"))
    return files[0] if files else None

def find_afcpt_json(folder):
    files = glob.glob(os.path.join(folder, "*afcpt.json"))
    return files[0] if files else None

def extract_from_results_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    try:
        if 'gsm8k' in data["results"]:
            val = data["results"]["gsm8k"]["exact_match,flexible-extract"]
        elif 'humaneval' in data["results"]:
            val = data["results"]["humaneval"]["post_process_pass@1"]
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

def extract_summary(root="hierarchy_decoding", output="summary.csv"):
    rows = []
    for type_dir in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(type_dir): continue
        for task_dir in glob.glob(os.path.join(type_dir, "*")):
            for param_dir in glob.glob(os.path.join(task_dir, "*")):
                instruct_dir = os.path.join(param_dir, "__root__GSAI-ML__LLaDA-8B-Instruct")
                if not os.path.isdir(instruct_dir): continue
                try:
                    type_, task, length, block_length, threshold, low_threshold, remask_threshold = parse_dir_name(param_dir)
                except Exception as e:
                    continue
                results_json = find_results_json(instruct_dir)
                afcpt_json = find_afcpt_json(param_dir)

                exact_match_flex = extract_from_results_json(results_json) if results_json else None
                avg_calls, tps, tps_our = (extract_from_afcpt_json(afcpt_json) if afcpt_json else (None, None, None))

                rows.append({
                    "type": type_,
                    "task": task,
                    "length": length,
                    "block_length": block_length,
                    "threshold": threshold,
                    "low_threshold": low_threshold,
                    "remask_threshold": remask_threshold,
                    "exact_match,flexible-extract": exact_match_flex,
                    "average forward calls per token": avg_calls,
                    "tokens per second": tps,
                    "tokens per second our": tps_our,
                })

    # 写入CSV
    with open(output, "w", newline="") as csvfile:
        fieldnames = [
            "type", "task", "length", "block_length", "threshold", "low_threshold", "remask_threshold",
            "exact_match,flexible-extract",
            "average forward calls per token", "tokens per second", "tokens per second our"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Summary saved to {output}")


def main():
    args = parse_args()
    if args.type == 'process_results':
        process_results(args.res_dir)
    elif args.type == 'extract_summary':
        extract_summary(args.res_dir)
    elif args.type == 'process_and_summary':
        process_results(args.res_dir)
        extract_summary(args.res_dir)
    else:
        raise ValueError('unimplemented type')

if __name__ == '__main__':
    main()
    # process_results('/mnt/dllm/dulun.dl/dllm_decoding/evals_results/h20/hierarchy_decoding/remask/mbpp')
    # extract_summary('/mnt/dllm/dulun.dl/dllm_decoding/evals_results/h20/hierarchy_decoding/')
    

