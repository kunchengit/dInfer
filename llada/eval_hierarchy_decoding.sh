# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
# export CUDA_VISIBLE_DEVICES=0,1

decoding=origin # or herachical_fast_v2 or herachical_remasking
length=256 # generate length
block_length=32
steps=$((length / block_length))  # only useful when threshold is None
model_path='/root/GSAI-ML__LLaDA-8B-Instruct'  # You can change the model path to LLaDA-1.5 by setting model_path='/root/LLaDA-1.5_f695918f7f6432bf'
threshold=0.9
remask_threshold=0.001
low_threshold=0.5
machine=new_branch
# path to save eval results
output_dir=/mnt/dllm/dulun.dl/dllm_decoding/evals_results/${machine}/hierarchy_decoding

# hierarchy decoding 

task=gsm8k
num_fewshot=5
output_path=${output_dir}/${task}/${length}_${block_length}_${threshold}_${low_threshold}_${remask_threshold}
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},remask_threshold=${remask_threshold},factor=${factor},show_speed=True,save_dir=${output_path},decoding=${decoding} \
--output_path ${output_path} 

task=humaneval  # The humaneval dataset requires saving the generated results for post-processing, so the --log_sample option must be added.
output_path=${output_dir}/${decoding}/${task}/${length}_${block_length}_${threshold}_${low_threshold}_${remask_threshold}
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},remask_threshold=${remask_threshold},factor=${factor},show_speed=True,save_dir=${output_path},decoding=${decoding} \
--output_path ${output_path} --log_samples

# task=minerva_math
# num_fewshot=4
# output_path=${output_dir}/${decoding}/${task}/${length}_${block_length}_${threshold}_${low_threshold}_${remask_threshold}
# accelerate launch eval_llada.py --tasks ${task}  --num_fewshot ${num_fewshot}\
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},remask_threshold=${remask_threshold},factor=${factor},show_speed=True,save_dir=${output_path},decoding=${decoding} \
# --output_path ${output_path}

# task=mbpp
# num_fewshot=3
# output_path=${output_dir}/${decoding}/${task}/${length}_${block_length}_${threshold}_${low_threshold}_${remask_threshold}
# accelerate launch eval_llada.py --tasks ${task}  --num_fewshot ${num_fewshot}\
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},remask_threshold=${remask_threshold},factor=${factor},show_speed=True,save_dir=${output_path},decoding=${decoding} \
# --output_path ${output_path}

# task=bbh
# num_fewshot=3
# output_path=${output_dir}/${decoding}/${task}/${length}_${block_length}_${threshold}_${low_threshold}_${remask_threshold}
# accelerate launch eval_llada.py --tasks ${task}  --num_fewshot ${num_fewshot}\
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},low_threshold=${low_threshold},remask_threshold=${remask_threshold},factor=${factor},show_speed=True,save_dir=${output_path},decoding=${decoding} \
# --output_path ${output_path}

# Automatically process generate resutls and summary them into a .csv file
python merge_results.py --type process_and_summary --res_dir ${output_dir}