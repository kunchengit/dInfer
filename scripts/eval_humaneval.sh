export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1


decoding=origin # or herachical_fast_v2 or herachical_remasking
length=256 # generate length
block_length=32
steps=$((length / block_length))  # only useful when threshold is None
model_path='/root/LLaDA-1.5_f695918f7f6432bf'  # You can change the model path to LLaDA-1.5 by setting model_path='/root/LLaDA-1.5_f695918f7f6432bf or instruct: /root/GSAI-ML__LLaDA-8B-Instruct'
threshold=0.9
remask_threshold=0.001
low_threshold=0.5


task=humaneval
time_stamp=$(date +%s)

if [[ "${model_path}" == *"LLaDA-1.5"* ]]; then
  model="LLaDA-1.5"
elif [[ "${model_path}" == *"LLaDA-8B-Instruct"* ]]; then
  model="LLaDA-8B-Instruct"
elif [[ "${model_path}" == *"LLaDA-8B-Base"* ]]; then
  model="LLaDA-8B-Base"
else
  model="LLaDA-unknown-version"
fi

output_dir=/mnt/dllm/dulun.dl/dllm_decoding/evals_results/${task}/${model}/genlen${length}/blk${block_length}/${decoding}/${time_stamp}
cd ../llada

accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},remask_threshold=${remask_threshold},factor=${factor},show_speed=True,save_dir=${output_path},decoding=${decoding} \
--output_path ${output_path} --log_samples
