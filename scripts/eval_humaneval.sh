decoding=origin # or herachical_fast_v2 or herachical_remasking
length=256 # generate length
block_length=32
steps=$((length / block_length))  # only useful when threshold is None
model_path='/root/LLaDA-1.5_f695918f7f6432bf'  # You can change the model path to LLaDA-1.5 by setting model_path='/root/LLaDA-1.5_f695918f7f6432bf or instruct: /root/GSAI-ML__LLaDA-8B-Instruct'
threshold=0.9
remask_threshold=0.001
low_threshold=0.5


echo yes > /mnt/dllm/dulun.dl/dllm_decoding/test.log



