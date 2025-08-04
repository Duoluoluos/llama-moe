import os
import json
import random
import argparse
from pathlib import Path

def load_json_file(file_path):
    """加载JSON文件数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: {file_path} is not a valid JSON file. Skipping...")
        return []

def save_json_file(data, output_path):
    """保存数据到JSON文件"""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Successfully saved {len(data)} samples to {output_path}")

def merge_json_files(input_dir, output_path, sample_percent=100):
    """从多个JSON文件中按比例采样并合并成一个新文件"""
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist")
    
    # 获取输入目录中的所有JSON文件
    json_files = list(Path(input_dir).glob('*.json'))
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # 从每个文件中采样数据
    all_samples = []
    for file_path in json_files:
        data = load_json_file(file_path)
        if not data:
            continue
        
        # 计算采样数量
        sample_count = int(len(data) * sample_percent / 100)
        sample_count = max(1, sample_count)  # 至少采样1条数据
        
        # 采样数据
        sampled_data = random.sample(data, sample_count)
        all_samples.extend(sampled_data)
        
        print(f"Sampled {sample_count}/{len(data)} samples from {file_path}")
    
    # 打乱所有采样数据
    random.shuffle(all_samples)
    
    # 保存合并后的数据
    save_json_file(all_samples, output_path)
    
    return all_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge JSON files with sampling')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing JSON files')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for merged JSON file')
    parser.add_argument('--sample_percent', type=int, default=100, help='Percentage of data to sample from each file (default: 100)')
    
    args = parser.parse_args()
    
    # 检查参数有效性
    if args.sample_percent < 1 or args.sample_percent > 100:
        raise ValueError('Sample percentage must be between 1 and 100')
    
    # 执行合并操作
    merge_json_files(args.input_dir, args.output_path, args.sample_percent)