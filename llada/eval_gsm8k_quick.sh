export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=256
block_length=32
num_fewshot=5
steps=$((length / block_length))
factor=1.0
# model_path='GSAI-ML/LLaDA-8B-Instruct'
# You can change the model path to LLaDA-1.5 by setting model_path='GSAI-ML/LLaDA-1.5'
model_path='/root/LLaDA-1.5_f695918f7f6432bf/'
output_path='/ossfs/workspace/res'


# baseline
accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot}  --output_path ${output_path} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${length},block_length=${block_length},show_speed=True

# parallel
#accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot}  --output_path ${output_path} \
#--confirm_run_unsafe_code --model llada_dist \
#--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.9,show_speed=True