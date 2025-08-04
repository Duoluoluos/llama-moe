#!/usr/bin/bash

set -vx

tokenizer_dir=/hdd-cifs/wangqi/models/Qwen2.5-1.5B
data_type=colab_data
data_dir=/hdd-cifs/wangqi/meld_data/calib_data
out_dir=/hdd-cifs/wangqi/meld_data/calib_data
logs_dir=logs
content_column=text



log_path=logs/tokenize_${data_type}.log
python -m smoe.utils.tokenize \
    -f jsonl \
    -t $tokenizer_dir \
    -c $content_column \
    -i $data_dir \
    -o $out_dir \
1>${log_path} 2>&1 &
echo "$data_type > $log_path"
