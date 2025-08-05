# file: qwen_rope_autoalign.py   （随便放，但必须在第一次 import transformers 前执行）

from transformers.models.qwen2 import modeling_qwen2 as qmod
import torch
from functools import wraps

_orig_apply = qmod.apply_rotary_pos_emb

@wraps(_orig_apply)
def apply_rotary_pos_emb_autoalign(q, k, cos, sin):
    """
    1. 若最后一维(head_dim)不对 => 转置 cos/sin
    2. 调整前导 1，使 cos.dim() == q.dim()-1  (官方稍后会再 unsqueeze(0))
    """
    # ---- 方向校正 ----
    if cos.shape[-1] != q.shape[-1]:
        cos, sin = cos.transpose(-2, -1), sin.transpose(-2, -1)

    # ---- 维度补/裁 ----
    target_dim = q.dim() - 1          # 官方 forward 里再 unsqueeze(0)
    while cos.dim() > target_dim:     # 多余 1 维 -> squeeze
        if cos.size(0) != 1:
            raise RuntimeError("Unexpected leading dim in cos/sin")
        cos, sin = cos.squeeze(0), sin.squeeze(0)
    while cos.dim() < target_dim:     # 缺维 -> unsqueeze
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)

    return _orig_apply(q, k, cos, sin)

# 注入
qmod.apply_rotary_pos_emb = apply_rotary_pos_emb_autoalign
