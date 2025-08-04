import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_ffn_hook_activations(model, tokenizer, texts, layer_idx=23, after_activation=True, device="cuda"):
    """
    返回指定 FFN 层的激活表示，用于 CKA.
    
    Args:
        model/tokenizer: 已加载的 Qwen2.* 模型与 tokenizer
        texts: List[str]，共享输入
        layer_idx: 目标 Transformer block 序号（0‑based）
        after_activation: True 取 mlp.act_fn 输出，否则取线性输出
    Returns:
        reps: Tensor [n_tokens, hidden_dim]
    """
    model.to(device).eval()
    # 1) 编码
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    
    # 2) 定义 hook
    activations = []
    def hook_fn(module, input, output):
        # output 形状 [batch, seq_len, hidden_dim]
        activations.append(output.detach().cpu())
    
    # 3) 根据 Qwen2 的层命名找到 FFN
    # Qwen2 框架: model.layers[i].mlp
    ffn_layer = model.model.layers[layer_idx].mlp
    handle = ffn_layer.act_fn.register_forward_hook(hook_fn) if after_activation \
             else ffn_layer.down_proj.register_forward_hook(hook_fn)  # 线性输出
    
    # 4) forward
    with torch.no_grad():
        model(**batch)
    
    # 5) 清理 hook 并返回
    handle.remove()
    reps = torch.cat(activations, dim=0)      # [batch, seq_len, hidden]
    reps = reps.reshape(-1, reps.size(-1))    # [n_tokens, hidden]
    return reps

def test():
    texts = ["def quick_sort(arr):", "Hello, how are you?"]   # 你的校准/评估文本
    target_layer = 20                                         # 选要上采样的层

    model_paths = [
        "/path/qwen2-7b",
        "/path/qwen2-7b-instruct",
        "/path/qwen2.5-7b-code",
        "/path/qwen2.5-7b-chat",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])  # 同架构共用 tokenizer
    reps_list = []

    for ckpt in model_paths:
        m = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
        reps = get_ffn_hook_activations(m, tokenizer, texts, layer_idx=target_layer)
        reps_list.append(reps)   # 后面 pair‑wise 做 CKA
        del m; torch.cuda.empty_cache()

def visualize_cka():
    # --- Data (same as yours) ---
    labels_en = ['E1: General', 'E2: Math', 'E3: Code', 'E4: Science']

    data_baseline1 = np.array([
        [1.00, 0.12, 0.15, 0.18], 
        [0.12, 1.00, 0.25, 0.30],
        [0.15, 0.25, 1.00, 0.20], 
        [0.18, 0.30, 0.20, 1.00]
    ])

    data_baseline2 = np.array([
        [1.00, 0.58, 0.65, 0.60], 
        [0.58, 1.00, 0.72, 0.75],
        [0.65, 0.72, 1.00, 0.68], 
        [0.60, 0.75, 0.68, 1.00]
    ])

    data_our_method = np.array([
        [1.00, 0.21, 0.26, 0.31], 
        [0.21, 1.00, 0.28, 0.37],
        [0.26, 0.28, 1.00, 0.35], 
        [0.31, 0.37, 0.35, 1.00]
    ])

    # --- Plotting Setup ---
    plt.rcParams['font.family'] = 'Liberation Serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Create a figure
    fig = plt.figure(figsize=(18, 5.5))
    fig.suptitle('Analysis of Inter-Expert Functional Specialization using CKA', fontsize=20, y=1.02)

    # **Create a GridSpec layout: 3 for plots, 1 for color bar**
    # The color bar column will be 5% of the width of a main plot column.
    gs = gridspec.GridSpec(1, 4, width_ratios=[20, 20, 20, 1])

    # Create axes from the GridSpec
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    cbar_ax = fig.add_subplot(gs[0, 3])

    # Manually handle shared y-axis
    ax1.get_yaxis().set_ticks([])
    ax2.get_yaxis().set_ticks([])

    # Common settings
    cmap = sns.color_palette("mako_r", as_cmap=True)
    heatmap_kws = {
        "annot": True, "fmt": ".2f", "cmap": cmap,
        "linewidths": .5, "annot_kws": {"size": 15.5},
        "vmin": 0, "vmax": 1
    }

    # --- Plotting Heatmaps ---

    # (a) Baseline
    sns.heatmap(data_baseline1, ax=ax0, **heatmap_kws, xticklabels=labels_en, yticklabels=labels_en, cbar=False)
    ax0.set_title('(a) Baseline: Original Experts', fontsize=15, pad=12)
    ax0.tick_params(axis='x', labelrotation=45, labelsize=13)
    ax0.tick_params(axis='y', labelrotation=0, labelsize=13)

    # (b) Naive Merging
    # **** 修正: 添加 yticklabels=False ****
    sns.heatmap(data_baseline2, ax=ax1, **heatmap_kws, xticklabels=labels_en, yticklabels=False, cbar=False)
    ax1.set_title('(b) Naive Upcycling', fontsize=15, pad=12)
    ax1.tick_params(axis='x', labelrotation=45, labelsize=13)

    # (c) Our Method - 将颜色条链接到此图
    # **** 修正: 添加 yticklabels=False ****
    sns.heatmap(data_our_method, ax=ax2, **heatmap_kws, xticklabels=labels_en, yticklabels=False, cbar=True, cbar_ax=cbar_ax)
    ax2.set_title('(c) Our Method (Functional Alignment)', fontsize=15, pad=12)
    ax2.tick_params(axis='x', labelrotation=45, labelsize=13)

    # Configure the color bar label
    cbar_ax.figure.axes[-1].set_ylabel('Centered Kernel Alignment (CKA) Score', rotation=270, labelpad=20, fontsize=14)
    
    # Use a tighter layout without the rect parameter, as GridSpec handles the spacing
    plt.tight_layout(pad=1.0)
    
    # Adjust for the main title
    fig.subplots_adjust(top=0.88) 

    plt.savefig("/home/wangqi/llama-moe/figs/fig_cka_comparison.pdf", format='pdf', bbox_inches='tight')
    plt.savefig("/home/wangqi/llama-moe/figs/fig_cka_comparison.png", format='png', bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    visualize_cka()