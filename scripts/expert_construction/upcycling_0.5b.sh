source /home/wangqi/miniconda3/etc/profile.d/conda.sh
conda activate smoe                       

workdir="/home/wangqi/llama-moe"
cd $workdir
# --- 基础配置 ---
# 定义构成专家的各个标准Llama模型的名称
# 第一个模型为锚点模型
declare -a expert_model_names=(
  "M1-0.5B"
  "M2-0.5B"
  "M3-0.5B"
  "M4-0.5B"
)

# --- MoE 和 门控网络配置 ---
num_selects=2             # 每个token选择的专家数
use_random_gate="True"   # 是否使用随机门控，"False"表示需要加载预训练权重
use_softmax="True"
multiply_gate_scores="True"
stage="merge"
score_scale_factor=1.0
template_path="/hdd-cifs/wangqi/models/Qwen2.5-0.5B"
# template_path="/hdd-cifs/wangqi/meld_models/Qwen-1.5B-times-3"
# --- 路径配置 ---
# 数据和模型的根目录
data_path="/hdd-cifs/wangqi/meld_models"
codata_path="/hdd-cifs/wangqi/meld_data/calib_data/E4-calib.jsonl"
# 构建模型路径列表
model_paths=()
for name in "${expert_model_names[@]}"; do
  model_paths+=("${data_path}/LoRA_Editted/${name}")
done

# 预训练门控权重的路径 (当 use_random_gate="False" 时必须提供)
num_experts=${#expert_model_names[@]} # 自动计算专家数量
select_file_path="${data_path}/gate_weights/${num_experts}Expert-Gate"

# 定义输出模型的保存路径
save_path_name="Qwen-0.5B-times-${num_experts}"
#save_path_name="Qwen-0.5B-times-${num_experts}-aligned"
save_path="${data_path}/${save_path_name}"

# --- 执行命令 ---
# 使用 python -m 执行你的脚本
# 注意 --model_paths 参数后面直接跟着由bash数组转换成的路径列表
echo "Starting MoE model merging..."
echo "Number of experts: ${num_experts}"
echo "Source models: ${model_paths[@]}"
echo "Saving to: ${save_path}"

python -m smoe.entrypoint.expert_construction.qwen_upcycle_dense \
  --template_path "${template_path}" \
  --model_paths "${model_paths[@]}" \
  --save_path "${save_path}" \
  --num_selects ${num_selects} \
  --use_random_gate ${use_random_gate} \
  --use_softmax ${use_softmax} \
  --multiply_gate_scores ${multiply_gate_scores} \
  --score_scale_factor ${score_scale_factor} \
  --stage ${stage} \
  --codata_path ${codata_path}
echo "Merging process finished."