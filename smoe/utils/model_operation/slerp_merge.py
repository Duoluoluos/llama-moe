import torch
import numpy as np

def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    epsilon = 1e-10

    # Convert tensors to a common format, float32
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)

    # Convert tensors to numpy arrays
    c = False
    if not isinstance(v0, np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1, np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles    
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)

    if norm_v0 > epsilon:
        v0 = v0 / norm_v0
    else:
        print(f"Warning: Norm of v0 is very small ({norm_v0}). Skipping normalization.")

    if norm_v1 > epsilon:
        v1 = v1 / norm_v1
    else:
        print(f"Warning: Norm of v1 is very small ({norm_v1}). Skipping normalization.")

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    del v0_copy, v1_copy
    del v1

    if c:
        res = torch.from_numpy(v2)
    else:
        res = v2
    return res


def linear_average(tensors: list, weights: list = None):
    """对多个张量进行线性平均"""
    if weights is None:
        weights = [1.0/len(tensors)] * len(tensors)
    total = torch.zeros_like(tensors[0])
    for tensor, weight in zip(tensors, weights):
        total += tensor * weight
    return total


def adaptive_merge(models: list, strategy: dict):
    """
    根据策略自适应地合并模型
    models: 模型状态字典列表
    strategy: {
        "attention": ("slerp", 0.5),  # (方法, 混合比例)
        "default": ("average", None)    # (方法, [各模型权重])
    }
    """
    merged_state = models[0].copy()
    num_models = len(models)
    
    # 遍历所有参数
    for key in merged_state.keys():
        # 跳过专家层参数
        if "mlp" in key or "calculator" in key or "expert" in key:
            continue
        # print(f"Processing parameter '{key}'")
        tensors = [model[key] for model in models if key in model]
        
        # 检查尺寸一致性
        if not all(tensor.shape == tensors[0].shape for tensor in tensors):
            continue
            
        method = strategy["default"][0] 
        params = strategy["default"][1]
        
        if any(attn_key in key for attn_key in ["q_proj", "k_proj", "v_proj", "o_proj", "attn"]):
            method = strategy["attention"][0]
            params = strategy["attention"][1]
        
        if method == "slerp" and len(tensors) == 2:
            merged_state[key] = slerp(params, tensors[0], tensors[1])
        elif method == "average":
            weights = params if params else [1/num_models] * num_models
            merged_state[key] = linear_average(tensors, weights)
    
    return merged_state