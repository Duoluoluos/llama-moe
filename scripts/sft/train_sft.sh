#!/bin/bash
source /home/wangqi/miniconda3/etc/profile.d/conda.sh   
conda activate smoe     
workdir="/home/wangqi/llama-moe"
cd $workdir
# 训练脚本：基于llamafactory库的SFT训练
# 使用方法: ./train_sft_llamafactory.sh [模型大小] [批次大小] [是否只训练gate]

# 配置参数
base_dir="/hdd-cifs/wangqi"
MODEL_PATH="${base_dir}/meld_models/Qwen-0.5B-times-4"
DATA_PATH="${base_dir}/meld_data/calib_data/calib_train.json"
OUTPUT_DIR="${base_dir}/meld_models/BTX_Qwen-0.5B-times-4"
tokenizer_name_or_path="${base_dir}/models/Qwen2.5-0.5B"
# accelerate launch smoe/entrypoint/sft/train_sft.py \
#     --model_type qwen \
#     --model_name_or_path $MODEL_PATH \
#     --dataset_dir_or_path $DATA_PATH \
#     --output_dir $OUTPUT_DIR \
#     --model_max_length 2048 \
#     --num_train_epochs 6 \
#     --tokenizer_name_or_path $tokenizer_name_or_path \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 4 \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 2 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --train_only_gate True \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --bf16 True \
#     --gradient_checkpointing True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
torchrun --standalone --nproc_per_node 8 \
smoe/entrypoint/sft/train_sft.py \
    --model_type qwen \
    --model_name_or_path $MODEL_PATH \
    --dataset_dir_or_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --model_max_length 2048 \
    --num_train_epochs 6 \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --do_train False \
    --train_only_gate True \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --gradient_checkpointing True