#!/bin/bash

# 训练脚本：基于llamafactory库的SFT训练
# 使用方法: ./train_sft_llamafactory.sh [模型大小] [批次大小] [是否只训练gate]

# 配置参数
base_dir=$(dirname $(dirname $(realpath $0)))
model_path="${base_dir}/models/llama-7b"
dataset_path="${base_dir}/data/merged"
output_dir="${base_dir}/outputs/sft_llamafactory"
model_max_length=2048
epochs=3
learning_rate=2e-5
warmup_ratio=0.03
gradient_accumulation_steps=1
train_only_gate=false

# 解析命令行参数
if [ $# -gt 0 ]; then
    model_size=$1
    case $model_size in
        "7b")
            model_path="${base_dir}/models/llama-7b"
            ;; 
        "13b")
            model_path="${base_dir}/models/llama-13b"
            ;; 
        *)
            echo "未知的模型大小: $model_size，使用默认值"
            ;; 
    esac
fi

if [ $# -gt 1 ]; then
batch_size=$2
else
batch_size=4
fi

if [ $# -gt 2 ] && [ "$3" = "true" ]; then
train_only_gate=true
fi

# 显示配置信息
echo "===== 训练配置 =====