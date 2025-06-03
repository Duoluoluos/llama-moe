#!/usr/bin/bash

#  llama_7B  llama_13B  llama_30B  llama_base
#  llama2_7B  llama2_13B  llama2_30B  llama2_base
#  open_llama_7b
#  Mistral-7B-v0.1
#  ReluLLaMA-7B
llama_size="Qwen2.5-7B-Instruct"

num_experts=16 #  8  16

data_path=/data/wangqi
model_path=${data_path}/models/${llama_size}
save_path=${data_path}/moefication_results/split


python -m smoe.entrypoint.expert_construction.llama_split_random \
  --model_path ${model_path} \
  --save_path ${save_path} \
  --template layers.{}.mlp.gate_proj.weight \
  --num_experts ${num_experts}
