import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
def ablation_bar():
    # --- 创建一个包含两个子图的画布 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5)) # 稍微调整高度以适应

    # --- 左侧: 柱状图 (a) ---
    labels = ['General ($M_1$)', 'Math ($M_2$)', 'Code ($M_3$)', 'Science ($M_4$)']
    id_scores_bar = [44.1, 40.8, 34.2, 37.5]
    ood_scores_bar = [35.3, 33.2, 26.1, 28.8]

    x_bar = np.arange(len(labels))
    width = 0.3

    ax1.bar(x_bar - width/2, id_scores_bar, width, label='Average In-Distribution (ID) Score', color='#4c5c9c')
    ax1.bar(x_bar + width/2, ood_scores_bar, width, label='Out-of-Distribution (OOD) Score', color='#6fbf9c')

    ax1.set_ylabel('Score', fontsize=16)
    # 将 (a) 添加到 x 轴标签
    ax1.set_xlabel('(a) Anchor Model', fontsize=16)
    ax1.set_xticks(x_bar)
    ax1.set_xticklabels(labels, fontsize=12, rotation=15)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.set_ylim(10, 48)
    
    # --- 右侧: 折线图 (b) ---
    num_experts = [1, 2, 3, 4]
    id_scores_line = [40.95, 42.83, 43.61, 44.07]
    ood_scores_line = [30.39, 33.85, 34.50, 35.26]

    ax2.plot(num_experts, id_scores_line, marker='o', linestyle='-', color='#4c5c9c')
    ax2.plot(num_experts, ood_scores_line, marker='s', linestyle='--', color='#6fbf9c')
    
    ax2.set_ylabel('Score', fontsize=16)
    # 将 (b) 添加到 x 轴标签
    ax2.set_xlabel('(b) Num of Experts', fontsize=16)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.set_xticks(num_experts)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(30, 45)

    # --- 统一的图例 ---
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, 
        labels, 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.02), # 稍微向上调整图例位置
        ncol=2, 
        fontsize=14,
        frameon=False
    )

    # --- 调整整体布局并保存 ---
    fig.tight_layout(rect=[0, 0, 1, 0.93]) # 调整 rect 确保图例不重叠
    plt.savefig("figs/fig_num_anchor.pdf", dpi=300)
    plt.savefig("figs/fig_num_anchor.png", dpi=300)


def show_discrepancy():
    fig, axes = plt.subplots(figsize=(6, 4))
    # 生成模拟数据
    data = {
        'w/o rewriting': [0.182, 0.245, 0.366],
        'rewriting w/o CPT': [0.227, 0.269, 0.370],
        'rewriting w/ CPT': [0.256,  0.295 ,0.389]
    }
    index = ['SRCQA', 'FintextQA', 'SyllabusQA']
    df = pd.DataFrame(data, index=index)

    # 绘制柱状图
    df = df.reset_index()
    df = pd.melt(df, id_vars=['index'], var_name='rewriting_type', value_name='Value')

    sns.barplot(x='index', y='Value', hue='rewriting_type', data=df, ax=axes, palette='deep')
    axes.set_ylim(0.15, 0.5)
    axes.set_title('Query-Document Discrepancy', fontsize=16)
    axes.set_ylabel('Semantic Similarity', fontsize=14)
    axes.legend(loc='upper left')
    axes.set_xlabel('', fontsize=16)

    # Adjusting layout
    plt.tight_layout()
    plt.savefig("qd_discrepancy.png", dpi=300)
    # Show plot
    plt.show()



def corpus_size_lineplot():
    # Sample data to illustrate the phenomenon
    tokens_proportion = np.array([ 0.2, 0.4, 0.6, 0.8, 1])
    srcqa_performance = np.array([0.541, 0.559, 0.588, 0.613 ,0.622])
    syllabusqa_performance = np.array([0.501, 0.507, 0.501, 0.513 ,0.517])
    fintextqa_performance = np.array([0.494, 0.492, 0.496, 0.500, 0.505])
    # Creating a single figure with two subplots
    fig, axes = plt.subplots(figsize=(6, 4))
    # Plotting for SRCQA
    sns.lineplot(x=tokens_proportion, y=srcqa_performance, marker='o', color='b', ax=axes, alpha=0.7, markersize=8, label='SRCQA')
    # Plotting for SyllabusQA
    sns.lineplot(x=tokens_proportion, y=syllabusqa_performance, marker='s', color='r', ax=axes, alpha=0.7, markersize=8, label='SyllabusQA')
    # Plotting for FintextQA
    sns.lineplot(x=tokens_proportion, y=fintextqa_performance, marker='^', color='g', ax=axes, alpha=0.7, markersize=8, label='FintextQA')
    # Set axis labels
    axes.set_xlabel("Proportion of Document Tokens used by CPT", fontsize=14)
    axes.set_ylabel("Accuracy/F1 Score", fontsize=14)
    # Add grid
    axes.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Add legend
    axes.legend()
    # Set title
    axes.set_title('Effects of CPT Corpus Size Variation', fontsize=16)
    # Adjusting layout
    plt.tight_layout()
    plt.savefig("corpus_size_line.png", dpi=300)
    # Show plot
    plt.show()


def draw_performance_curve():
    # 数据
    parameter_scale = np.array([0.5, 1.5, 3, 7])
    accuracy = np.array([0.40, 0.49,  0.56, 0.62])
    training_time = np.array([21.97, 14.87, 31.60, 43.9])

    # 设置Seaborn样式
    sns.set(style="whitegrid", palette="muted", font="Times New Roman", font_scale=1.2)

    # 设置图形尺寸，适合双栏论文
    plt.figure(figsize=(6, 2), dpi=300)

    # 创建图形和坐标轴
    fig, ax1 = plt.subplots(dpi=300)

    # 绘制Accuracy曲线
    color = sns.color_palette("Reds")[2]  # 选择红色调的颜色
    ax1.set_xlabel('Parameter Scale (B)', fontsize=14)
    ax1.set_ylabel('Acc',  fontsize=14)
    ax1.set_ylim(0.2, 0.65)
    ax1.plot(parameter_scale, accuracy, 'o-', color=color, label='Accuracy', markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False)
    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制Training Time曲线
    color = sns.color_palette("Blues")[2]  # 选择蓝色调的颜色
    ax2.set_ylabel('Training Time for 100k Tokens\nUsing a Single Nvidia 4090 GPU (s)', fontsize=14)    
    ax2.set_ylim(12, 90)
    ax2.plot(parameter_scale, training_time, 'o-', color=color, label='Training Time', markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True, linestyle='--', alpha=0.5)
    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='lower right', ncol=1, frameon=True)
    # 设置标题
    plt.title('Impact of Parameter Scaling', fontsize=16)
    # 调整图形布局
    fig.tight_layout()

    # 添加网格
    #ax1.grid(True, linestyle='--', alpha=0.7)
    # 显示图形
    plt.show()

def draw_k_performance_curve():
    # 数据
    k_values = [1, 2, 4, 5, 6, 8, 10]
    accuracy = [0.524, 0.568, 0.622, 0.638, 0.641, 0.635, 0.622]

    # 找到峰值点
    peak_index = accuracy.index(max(accuracy))
    peak_k = k_values[peak_index]
    peak_acc = accuracy[peak_index]

    # 绘制图形
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(k_values, accuracy, marker='o', linestyle='-', color='tab:blue', linewidth=1.5, markersize=6)

    # 标注峰值点
    plt.annotate(f'Peak (k={peak_k}, {peak_acc:.3f})', 
                xy=(peak_k, peak_acc), 
                xytext=(peak_k + 0.5, peak_acc - 0.02), 
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)

    # 设置坐标轴
    plt.xlabel('Number of Retrieved Documents ($k$)', fontsize=12)
    plt.ylabel('Accuracy (SRCQA)', fontsize=12)
    plt.xticks(k_values, rotation=45)  # x轴标签旋转45度
    plt.yticks([0.5, 0.55, 0.6, 0.65])
    plt.grid(linestyle='--', alpha=0.7, color='lightgray')

    # 添加网格和边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # 保存为 PDF（用于 LaTeX 插入）
    plt.savefig('k_accuracy_plot.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Liberation Serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    ablation_bar()