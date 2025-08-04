from aistudio_common.openapi import DataStore
from pypai.conf import ExecConf, KMConf, CodeRepoConf
from pypai.job import PythonJobBuilder
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
YAML_DIR = CURRENT_DIR.parent / 'yamls'


def submit_single_job(script_file, yaml_file, mount_command, **kwargs):
    """
    提交单个模型的评测任务
    """
    # 获取模型名
    # model_name = os.path.basename(os.path.normpath(model_path))

    #dataset_str = datasets
    # 默认配置提取
    app_name = kwargs.get('app_name', 'graphhuanan')
    gpu_type = kwargs.get('gpu_type', 'h20-3e')
    priority = kwargs.get('priority', 'high')
    gpu_num = kwargs.get('gpu_num', 8)
    pod_num = kwargs.get('pod_num', 1)
    max_workers = kwargs.get('max_workers', 8)
    branch = kwargs.get('branch', 'master')

    image = kwargs.get("image", "reg.docker.alibaba-inc.com/aii/aistudio:12910142-20250729192318")
    repo_url = kwargs.get("repo_url", "https://code.alipay.com/dulun.dl/Fast-dllm")
    
    repo_conf = CodeRepoConf(repo_url=repo_url, branch=branch)

    if gpu_type == 'h20-3e':
      memory = 196008 * gpu_num #192GB per GPU
    else: 
      memory = 102400 * gpu_num # 100GB per GPU

    workdir = r"/workspace/bin/Fast-dllm"
    homedir = r"/workspace/bin"
    script_dir = os.path.join (workdir, "scripts")
    llada_dir = os.path.join (workdir, "llada")
    yaml_dir = os.path.join (workdir, "yamls")
    local_yaml_dir = str(YAML_DIR)

    base_user_command = [
      r"mkdir /mnt/dllm && ",
      f"{mount_command} && ",
      r"cp /mnt/dllm/dulun.dl/dllm_decoding/huggingface.zip ~/.cache && ",
      r"cd ~/.cache && ",
      r"unzip huggingface.zip && ",
      r"rm -rf huggingface.zip && " 
    ]

    if yaml_file is None:
      base_user_command.extend([f"cd {script_dir} && ", f"bash {script_file}"])
    else:
      base_user_command.extend([
        f"mv -f {homedir}/*.yaml {yaml_dir}/ && ",
        f"cd {llada_dir} && ", 
        f"python eval_llada_yaml.py -y {yaml_file}"])
    # current dir: /workspace/bin
    full_user_command = "".join(base_user_command)

    job = PythonJobBuilder(
        code_repo_configs=[repo_conf],
        source_root=local_yaml_dir,
        main_file="",
        km_conf=KMConf(image=image),
        command=full_user_command,
        k8s_app_name=app_name,
        k8s_priority=priority,
        worker=ExecConf(
            cpu=16 * gpu_num,
            memory=memory,
            gpu_num=gpu_num,
            num=pod_num,
            gpu_type=gpu_type
        ),
        data_stores="",
        tag=f"",
        host_network=True,
        rdma=True,
        enable_pcache_fuse=True,
    )
    
    try:
      record_id = job.run()
      print(f"Submitted job!, record_id: {record_id}")
    except Exception as e:
      print (f"Failed to sumbit job: {e}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument('--script', type=str, help='Which script you want to run?')
  parser.add_argument('--yaml', type=str, help='Which yaml you want to use?')
  parser.add_argument('--mount', type=str, required = True, help='mount dllm command')

  args = parser.parse_args()

  if not ((args.script is None) ^ (args.yaml is None)):
    raise TypeError ("Please provide script or yaml parameters")
  
  submit_single_job (args.script, args.yaml, args.mount)