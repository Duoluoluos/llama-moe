#!/usr/bin/bash

set -vx

tokenizer_dir=/data/wangqi/models/Qwen2.5-7B-Instruct
data_type=toy_cpt_ds
data_dir=/data/wangqi/code
out_dir=/data/wangqi/code_tokenized
logs_dir=logs
content_column=text

mkdir -p $out_dir
mkdir -p $logs_dir


log_path=logs/tokenize_${data_type}_toy.log
python -m smoe.utils.tokenize \
    -f jsonl \
    -t $tokenizer_dir \
    -c $content_column \
    -i $data_dir \
    -o $out_dir \
1>${log_path} 2>&1 &
echo "$data_type > $log_path"
