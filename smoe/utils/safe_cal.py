# utils/safe_expert.py
import math, torch, torch.nn.functional as F
from smoe.modules.moe.moe_experts import LinearGLUExperts
from smoe.utils.debugging import value_print
CLIP_W = 1.0e4     # 权重幅值上限
CLIP_Y = 1.0e4     # 输出幅值上限

# ---------- 重写 reset_parameters：按 1/√num_experts 缩放 ----------
def _safe_reset(self: LinearGLUExperts):
    gain = 1.0 / math.sqrt(self.num_experts)
    for i in range(self.num_experts):
        for w in (self.weight_gate[i], self.weight_up[i], self.weight_down[i]):
            torch.nn.init.xavier_uniform_(w, gain=gain)
        if self.bias_gate is not None:
            torch.nn.init.zeros_(self.bias_gate[i])
            torch.nn.init.zeros_(self.bias_up[i])
            torch.nn.init.zeros_(self.bias_down[i])

# ---------- 重写 forward ----------
@torch.no_grad()                      # 只做数值检查，不影响梯度
def _check_weight_finite(w, name):
    if not torch.isfinite(w).all():
        raise RuntimeError(f"{name} contains NaN/Inf!")

def _safe_forward(self: LinearGLUExperts, x: torch.Tensor, idx: int):
    # a. 取出并检查权重
    w_gate = self.weight_gate[idx]
    w_up   = self.weight_up[idx]
    w_down = self.weight_down[idx]
    # value_print("w_gate", w_gate)
    # value_print("w_up", w_up)
    # value_print("w_down", w_down)
    _check_weight_finite(w_gate, f"w_gate[{idx}]")
    _check_weight_finite(w_up,   f"w_up[{idx}]")
    _check_weight_finite(w_down, f"w_down[{idx}]")

    # b. 将权重&输入转 fp32 并 clip，防溢出
    w_gate_f = w_gate.float().clamp_(-CLIP_W, CLIP_W)
    w_up_f   = w_up.float().clamp_(-CLIP_W, CLIP_W)
    w_down_f = w_down.float().clamp_(-CLIP_W, CLIP_W)
    x_f      = x.float()

    b_gate = None if self.bias_gate is None else self.bias_gate[idx].float().clamp_(-CLIP_W, CLIP_W)
    b_up   = None if self.bias_up   is None else self.bias_up[idx].float().clamp_(-CLIP_W, CLIP_W)
    b_down = None if self.bias_down is None else self.bias_down[idx].float().clamp_(-CLIP_W, CLIP_W)

    # c. 计算
    gate = self.act_fn(F.linear(x_f, w_gate_f, b_gate))
    up   =                F.linear(x_f, w_up_f,   b_up)
    prod = gate * up
    down =               F.linear(prod, w_down_f, b_down)

    # d. 输出 clip 后 cast 回原 dtype
    return down.clamp_(-CLIP_Y, CLIP_Y).to(x.dtype)


