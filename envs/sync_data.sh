#!/bin/bash
# sync_data.sh

NODE24="wangqi@100.64.1.24"  # Paoding B10
CODE_DIR="/home/wangqi/llama-moe"    # 代码目录
DATA_DIR="/data/wangqi"    # 数据目录
ENV_DIR="/home/wangqi/miniconda3"
BASH_FILE="/home/wangqi/.bashrc"
# 同步代码
rsync -avz --delete --exclude='.git' --exclude='__pycache__' --exclude='outputs' \
    -e ssh $CODE_DIR/ $NODE24:$CODE_DIR/

# 同步数据（首次需要全量同步，后续可增量）
rsync -avz --partial --progress -e ssh $DATA_DIR/ $NODE24:$DATA_DIR/
rsync -avz --partial --progress -e ssh $ENV_DIR/ $NODE24:$ENV_DIR/
rsync -avz --partial --progress -e ssh $BASH_FILE $NODE24:$BASH_FILE
