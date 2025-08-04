# SFT训练代码使用指南

## 概述
本目录包含基于原始Transformers库实现的SFT（监督微调）训练代码，支持两种数据格式，并提供了数据合并工具。

## 文件结构
- `train_sft.py`: 主训练脚本，支持两种数据格式
- `scripts/data_processing/merge_json_files.py`: JSON文件合并工具（Python脚本）
- `scripts/data_processing/merge_json_files.sh`: JSON文件合并工具（Shell脚本）

## 数据集格式支持
训练脚本支持以下两种数据格式：

### 格式1：对话格式（原始格式）
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "你好"
    },
    {
      "from": "gpt",
      "value": "你好！我是一个AI助手。"
    }
  ],
  "system_prompt": "你是一个有用的AI助手。"
}
```

### 格式2：指令-输入-输出格式
```json
{
  "instruction": "将下面的文本翻译成英文",
  "input": "你好，世界！",
  "output": "Hello, world!",
  "system_prompt": "你是一个翻译助手。"
}
```
- `instruction`: 指令文本
- `input`: 输入文本（可选）
- `output`: 输出文本
- `system_prompt`: 系统提示（可选）

## JSON文件合并工具使用方法
合并工具用于从多个JSON文件中按比例采样数据并合并成一个新文件。

### 使用Shell脚本
```bash
cd /home/wangqi/llama-moe
chmod +x scripts/sft/merge_json_files.sh
scripts/data_processing/merge_json_files.sh --input_dir <输入目录> --output_path <输出路径> [--sample_percent <采样百分比>]
```
例如：
```bash
scripts/data_processing/merge_json_files.sh --input_dir data/raw --output_path data/merged/train.json --sample_percent 50
```

### 使用Python脚本
```bash
cd /home/wangqi/llama-moe
python3 scripts/data_processing/merge_json_files.py --input_dir <输入目录> --output_path <输出路径> --sample_percent <采样百分比>
```
例如：
```bash
python3 scripts/data_processing/merge_json_files.py --input_dir data/raw --output_path data/merged/train.json --sample_percent 50
```

## 运行训练
### 安装依赖
```bash
pip install -r requirements.txt
```

### 准备数据
1. 将原始JSON文件放入一个目录（如`data/raw`）
2. 使用合并工具将数据合并：
   ```bash
   scripts/sft/merge_json_files.sh --input_dir data/raw --output_path data/merged/train.json --sample_percent 100
   ```

### 运行训练脚本
```bash
python3 smoe/entrypoint/sft/train_sft.py \
    --model_name_or_path <模型路径> \
    --dataset_dir_or_path data/merged/train.json \
    --output_dir <输出目录> \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --model_max_length 2048 \
    --train_only_gate <true/false> \
    --freeze_gate <true/false>
```

## 关键参数说明
- `--model_name_or_path`: 预训练模型路径
- `--dataset_dir_or_path`: 数据集路径
- `--output_dir`: 模型输出目录
- `--num_train_epochs`: 训练轮数
- `--per_device_train_batch_size`: 每个设备的批次大小
- `--model_max_length`: 最大序列长度
- `--train_only_gate`: 是否只训练gate部分
- `--freeze_gate`: 是否冻结gate部分

## 注意事项
1. 确保所有JSON文件格式正确，避免出现语法错误
2. 训练前建议先验证数据格式是否符合要求
3. 根据实际硬件情况调整批次大小和序列长度
4. 使用`train_only_gate`参数时，确保模型包含gate组件