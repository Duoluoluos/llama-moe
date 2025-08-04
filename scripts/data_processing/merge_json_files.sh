#!/bin/bash

# 脚本用法: ./merge_json_files.sh --input_dir <输入目录> --output_path <输出路径> --sample_percent <采样百分比>

# 显示帮助信息
show_help() {
    echo "Usage: $0 --input_dir <input_directory> --output_path <output_file_path> [--sample_percent <percentage>]"
    echo ""
    echo "Options:"
    echo "  --input_dir        包含JSON文件的输入目录 (必填)"
    echo "  --output_path      合并后的JSON文件输出路径 (必填)"
    echo "  --sample_percent   从每个文件采样的百分比 (可选，默认为100)"
    echo "  --help, -h         显示帮助信息"
    exit 1
}

# 默认参数值
SAMPLE_PERCENT=100

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --sample_percent)
            SAMPLE_PERCENT="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            ;;
    esac
    done

# 检查必填参数
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_PATH" ]]; then
    echo "Error: Missing required parameters"
    show_help
fi

# 检查采样百分比是否有效
if ! [[ "$SAMPLE_PERCENT" =~ ^[0-9]+$ ]] || [ "$SAMPLE_PERCENT" -lt 1 ] || [ "$SAMPLE_PERCENT" -gt 100 ]; then
    echo "Error: Sample percentage must be an integer between 1 and 100"
    exit 1
fi

# 检查输入目录是否存在
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

# 创建输出目录（如果不存在）
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# 运行Python脚本
echo "Merging JSON files from $INPUT_DIR with $SAMPLE_PERCENT% sampling..."
python3 "/home/wangqi/llama-moe/scripts/sft/merge_json_files.py" \
    --input_dir "$INPUT_DIR" \
    --output_path "$OUTPUT_PATH" \
    --sample_percent "$SAMPLE_PERCENT"

# 检查Python脚本是否成功执行
if [[ $? -eq 0 ]]; then
    echo "Merge completed successfully!"
else
    echo "Merge failed!"
    exit 1
fi