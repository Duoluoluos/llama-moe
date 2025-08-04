import os
import shutil
from collections import Counter
from tqdm import tqdm
import torch, torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM, AutoConfig
from smoe.models.qwen_moe.modeling_qwen_moe import (
    QwenMoEConfig,
    QwenMoEForCausalLM,
)
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Iterable
from smoe.utils.io import torch_load_template_file
from smoe.utils.model_operation.slerp_merge import adaptive_merge
from pathlib import Path
import json
import re

def upcycle_qwen_models_to_moe(
    template_path,
    model_paths,
    save_path,
    num_selects,
    score_scale_factor,
    use_random_gate=True,
    slerp_ratio=0.7,         
    **kwargs, 
):
    num_experts = len(model_paths)
    print(f"Starting merge of {num_experts} Qwen2 models into one MoE model.")

    # --- 1. 加载基础模型和配置，并反推真实维度 ---
    print("Step 1: Loading base model and inferring true dimensions from weights...")
    base_model_state_dict = AutoModelForCausalLM.from_pretrained(template_path).state_dict()
    source_config = AutoConfig.from_pretrained(template_path)

    example_down_proj_key = "model.layers.0.mlp.down_proj.weight"
    if example_down_proj_key not in base_model_state_dict:
        example_down_proj_key = "model.layers.0.mlp.down_proj.base_layer.weight" # 兼容LoRA
    
    true_intermediate_size = base_model_state_dict[example_down_proj_key].shape[1]
    print(f"  - Inferred TRUE intermediate_size from weights: {true_intermediate_size}")
    
    # --- 2. 构建正确且纯净的MoE配置 ---
    print("Step 2: Building clean MoE config with correct dimensions...")
    config_dict = source_config.to_dict()
    config_dict["intermediate_size"] = true_intermediate_size  *  num_experts # 强制使用我们反推出的真实尺寸
    config_dict.update({
        "num_experts": num_experts,
        "num_selects": num_selects,
    })
    config_moe = QwenMoEConfig.from_dict(config_dict)

    # --- 3. 初始化空的、尺寸正确的MoE模型 ---
    print("Step 3: Initializing empty MoE model")
    model_moe = QwenMoEForCausalLM(config_moe)
    model_moe.to("cpu")

    # --- 4. 精确地构建最终的权重字典 ---
    print("Step 4: Building the final state dictionary")
    final_state_dict = {}

    # Part A: 填充非专家层 (从第一个模型)
    # 这里我打算将第n个模型与第1个模型的非专家层的权重进行合并，这样就不会丢失任何信息。
    print("  - Part A: Merging non-expert layers...")
    strategy = {
        "attention": ("slerp", slerp_ratio), 
        "default": ("average", None)
    }
    
    # 加载所有模型状态
    model_states = []
    for path in model_paths:
        model = AutoModelForCausalLM.from_pretrained(path)
        model_states.append(model.state_dict())
        # 应用自适应合并
        merged_base_state = adaptive_merge(model_states, strategy)

    non_expert_keys = [
        k for k in merged_base_state.keys() 
        if "mlp" not in k
    ]
    
    for key in tqdm(non_expert_keys, desc="Non-expert layers"):
        final_state_dict[key] = merged_base_state[key]

    # Part B: 逐一填充每个专家的MLP层
    print("  - Part B: Merging MLP layers as experts...")
    for expert_idx in tqdm(range(num_experts), desc="Merging experts"):
        expert_state_dict = model_states[expert_idx]
        for layer_idx in range(config_moe.num_hidden_layers):
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                source_key_std = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                source_key_peft = f"model.layers.{layer_idx}.mlp.{proj_name}.base_layer.weight"
                source_tensor = expert_state_dict.get(source_key_peft)
                if source_tensor is None:
                    source_tensor = expert_state_dict.get(source_key_std)
                
                if source_tensor is not None:
                    target_proj_name = proj_name.replace('_proj', '')
                    target_key = f"model.layers.{layer_idx}.mlp.calculator.experts.weight_{target_proj_name}.{expert_idx}"
                    final_state_dict[target_key] = source_tensor.cpu().clone()

    # Part C: Initialize gate weights
    print("  - Part C: Initializing gate weights...")
    for layer_idx in tqdm(range(config_moe.num_hidden_layers), desc="Initializing gates"):
        final_state_dict[f"model.layers.{layer_idx}.mlp.gate.gate_network.0.weight"] = torch.randn(num_experts, config_moe.hidden_size)
        final_state_dict[f"model.layers.{layer_idx}.mlp.gate.gate_network.2.weight"] = torch.randn(num_experts, num_experts)        
        final_state_dict[f"model.layers.{layer_idx}.mlp.gate.weight_noise.weight"] = torch.zeros(
            (num_experts, config_moe.hidden_size)
        )

    # --- 5. 加载最终权重并保存模型 ---
    print("Step 5: Loading the final state dictionary into the model")
    model_moe.load_state_dict(final_state_dict)
    model_moe.half()

    print("Step 6: Saving converted model...")
    config_moe.save_pretrained(save_path)
    model_moe.save_pretrained(save_path)
    print(f"Successfully merged {num_experts} models. MoE model saved to \"{save_path}\".")


def load_input_ids(jsonl_path: str, max_tokens: int = 4096) -> torch.Tensor:
    """读取 JSONL，累积 ≤ max_tokens token，返回 (1, L) LongTensor"""
    ids: List[int] = []
    with Path(jsonl_path).open() as f:
        for line in f:
            ids.extend(json.loads(line)["input_ids"])
            if len(ids) >= max_tokens:
                break
    return torch.tensor(ids[:max_tokens]).unsqueeze(0)   # (1, L)

def hungarian_perm(C: torch.Tensor) -> List[int]:
    """C: (m,m) 相似度 -> 列重排 perm"""
    r, c = linear_sum_assignment((-C).cpu().numpy())     # maximize
    return c.tolist()

def permute_expert_weights(sd: Dict[str, torch.Tensor],
                           base: str,
                           e: int,
                           perm: List[int]) -> None:
    """base = 'model.layers.{L}.mlp.calculator.experts'"""
    up_key   = f"{base}.weight_up.{e}"        # (m, d)
    down_key = f"{base}.weight_down.{e}"      # (d, m)
    bias_key = f"{base}.bias.{e}"             # (m, )
    sd[up_key]   = sd[up_key][perm, :]                        # 行重排
    sd[down_key] = sd[down_key][:, perm]                      # 列重排
    if bias_key in sd:
        sd[bias_key] = sd[bias_key][perm]                     # 行重排


def get_expert_indices(sd: Dict[str, torch.Tensor], base: str) -> List[int]:
    """返回当前层存在的专家 idx 列表，如 [0,1,2]"""
    pattern = re.compile(rf"{re.escape(base)}\.weight_up\.(\d+)")
    return sorted(int(m.group(1)) for k in sd.keys() if (m := pattern.match(k)))


def align_all_layers(template_path: str,
                     jsonl_path: str,
                     save_path: str,
                     ref_exp: int = 0,
                     max_tokens: int = 4096):
    print(">> Loading model...")
    cfg   = QwenMoEConfig.from_pretrained(template_path)
    model = QwenMoEForCausalLM.from_pretrained(template_path, torch_dtype=torch.float32)

    sd = model.state_dict()

    # 1) 共享 hidden states  H ∈ ℝ^{L×d}
    print(">> Preparing hidden states...")
    with torch.no_grad():
        input_ids = load_input_ids(jsonl_path, max_tokens)            # (1, L)
        emb_layer = model.get_input_embeddings()                      # 通用取 embedding
        H = emb_layer(input_ids).squeeze(0)                           # (L, d)  CPU float32

    # 2) 遍历所有层
    print(">> Aligning experts layer-by-layer...")
    for l_idx in range(cfg.num_hidden_layers):
        base = f"model.layers.{l_idx}.mlp.calculator.experts"
        expert_ids = get_expert_indices(sd, base)
        if not expert_ids:                   # 非 MoE 层
            continue

        if ref_exp not in expert_ids:
            raise ValueError(f"Layer {l_idx} 没有 expert {ref_exp}，"
                             f"实际专家集合：{expert_ids}")

        # 取 reference 专家
        W_up_ref = sd[f"{base}.weight_up.{ref_exp}"]           # (m, d)
        m, d = W_up_ref.shape
        Z_ref = torch.relu(H @ W_up_ref.T).T                   # (m, L)

        # 对齐其它专家
        for e in expert_ids:
            if e == ref_exp:
                continue
            W_up_e = sd[f"{base}.weight_up.{e}"]               # (m, d)
            Z_e    = torch.relu(H @ W_up_e.T).T                # (m, L)
            C      = Z_ref @ Z_e.T                             # (m, m)
            perm   = hungarian_perm(C)
            permute_expert_weights(sd, base, e, perm)

        print(f"  Layer {l_idx:02d}: aligned {len(expert_ids)} experts → {ref_exp}")

    print(">> Saving aligned checkpoint...")
    model.load_state_dict(sd, strict=False)
    model.save_pretrained(save_path)
    print("✅ Done!  Aligned model saved to", save_path)

