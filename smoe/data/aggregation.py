from itertools import chain
import json
import random
import os
from typing import List, Dict, Any
def process_jsonl_to_json(input_file_path, output_file_path):
    processed_data = []
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析每一行JSON数据
            data = json.loads(line.strip())
            
            # 提取所需字段
            problem = data.get('problem', '')
            solution = data.get('solution', '')
            
            # 构建新的JSON对象
            processed_item = {
                "instruction": "",
                "input": problem,
                "output": solution
            }
            
            processed_data.append(processed_item)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，已保存到 {output_file_path}")

    
def process_json(input_file_path, output_file_path):
    processed_data = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        # 解析每一行JSON数据
        data = json.load(f)
        for item in data:
            # 提取所需字段
            problem = item.get('query', '')
            solution = item.get('response', '')

            # 构建新的JSON对象
            processed_item = {
                "instruction": "",
                "input": problem,
                "output": solution
            }

            processed_data.append(processed_item)
    
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，已保存到 {output_file_path}")

    
def group_texts(examples: dict, block_size: int = 1024):
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def group_instances(examples: list[dict], block_size: int = 2048) -> list[dict]:
    """
    Concate examples to a length of block size.

    Args:
        examples: a list of dict instances that have multiple keys
        block_size: the length of the concatenated examples
    """

    def _concat(examples: list[dict]) -> dict:
        """
        Concatenate the values of each key in the examples.

        Args:
            examples: a list of dict instances that have multiple keys
        """
        concatenated_examples = {}
        keys = examples[0].keys()
        for k in keys:
            concatenated_examples[k] = list(chain(*[e[k] for e in examples]))
        if "labels" not in keys and "input_ids" in keys:
            concatenated_examples["labels"] = concatenated_examples["input_ids"]
        return concatenated_examples

    def _chunk(examples: dict, block_size: int) -> list[dict]:
        """
        Split the concatenated examples into chunks of block_size.

        Args:
            examples: a dict instance that has multiple keys
            block_size: the length of the concatenated examples
        """
        total_length = len(examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in examples.items()
        }
        return result

    def _decompose(example: dict) -> list[dict]:
        """
        Decompose the example into a list of dict instances.

        Args:
            example: a dict instance that has multiple keys
        """
        num_chunks = len(example[list(example.keys())[0]])
        return [{k: example[k][i] for k in example.keys()} for i in range(num_chunks)]

    concatenated_examples = _concat(examples)
    chunk = _chunk(concatenated_examples, block_size)
    return _decompose(chunk)


def agg_json2jsonl(file_paths: List[str], output_path: str, sample_ratio: float = 0.1) -> None:
    """
    处理多个JSON文件，每个文件按比例采样后转换为JSONL格式
    
    Args:
        file_paths: JSON文件路径列表
        output_path: 输出JSONL文件路径
        sample_ratio: 采样比例，默认为0.1 (10%)
    """
    all_items = []
    
    # 处理每个JSON文件
    for file_path in file_paths:
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据格式
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                print(f"警告: 文件 {file_path} 不是预期的字典列表格式，跳过处理")
                continue
            
            # 计算采样数量
            sample_size = max(1, int(len(data) * sample_ratio))  # 至少采样1条
            
            # 随机采样
            sampled_items = random.sample(data, sample_size)
            
            # 处理采样数据
            for item in sampled_items:
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")
                
                # 合并instruction和input，中间加换行符
                text = f"{instruction}\n{input_text}".strip()
                
                # 添加到结果列表
                all_items.append({"text": text})
        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 写入JSONL文件
    if all_items:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"成功生成JSONL文件: {output_path}，共 {len(all_items)} 条记录")
    else:
        print("没有有效数据生成JSONL文件")

if __name__ == "__main__":
    # 配置参数
    json_files = [
        "/hdd-cifs/wangqi/meld_data/M1-alpaca.json",
        "/hdd-cifs/wangqi/meld_data/M2-MetaMathQA.json",
        "/hdd-cifs/wangqi/meld_data/M3-Magicoder-OSS.json",
        "/hdd-cifs/wangqi/meld_data/M4-SciQAG.json"
    ]
    output_jsonl = "/hdd-cifs/wangqi/meld_data/calib_data/E4-calib.jsonl"
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
     # 处理文件
    agg_json2jsonl(json_files, output_jsonl, sample_ratio=0.1)    