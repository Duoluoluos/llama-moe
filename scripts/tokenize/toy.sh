#!/usr/bin/bash

set -vx

tokenizer_dir=/data/wangqi/models/Qwen2.5-7B-Instruct
data_dir=/data/wangqi/toy_cpt_ds/ja_cc
out_dir=/data/wangqi/toy_cpt_ds/ja_cc_tokenized
logs_dir=logs
content_column=text

mkdir -p $out_dir
mkdir -p $logs_dir

for data_type in $(ls $data_dir)
do
    log_path=logs/tokenize_${data_type}_jatoy.log
    python -m smoe.utils.tokenize \
        -f jsonl \
        -t $tokenizer_dir \
        -c $content_column \
        -i $data_dir/$data_type \
        -o $out_dir/$data_type \
    1>${log_path} 2>&1 &
    echo "$data_type > $log_path"
done