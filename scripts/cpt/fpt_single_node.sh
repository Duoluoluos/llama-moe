workdir="/home/wangqi/llama-moe"
cd $workdir
# 手动配置节点参数（原SLURM参数）
num_nodes=1          # 总节点数
num_gpu_per_node=8   # 每节点GPU数
job_name="cpt-v2-7b" # 任务名称


export MASTER_ADDR="localhost"
export MASTER_PORT=12355
export RANK=0
export WORLD_SIZE=1

# 环境变量配置
export OMP_NUM_THUM_THREADS=1
export LOGLEVEL=INFO
# export NCCL_DEBUG=INFO  # 需要时取消注释

##############################################################
### 模型/训练参数配置（保留原配置）###
model_type="llama_moe"
tokenizer_path=/data/wangqi/models/Qwen2.5-7B-Instruct
dataset_dir=/data/wangqi/toy_cpt_ds
validation_dir=/data/wangqi/code_val
pretrained_model=/data/wangqi/models/LlamaMoEForCausalLM/Random/Qwen2.5-7B-Instruct-16Select4-gate_proj-Scale1.0
lr=3e-6
final_lr_portion=0.1
per_device_train_batch_size=2
per_device_eval_batch_size=2
gradient_accumulation_steps=4
block_size=4096
# 注意：需要设置训练数据量（单位：tokens）
num_tokens="50000000000"
seed=1227
deepspeed_config_file=conf/deepspeed/bf16_zero1_default.json
num_selects=4
##############################################################

# 计算训练参数（保留原逻辑）
max_steps=$(echo "${num_tokens} / ($block_size * $per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node)" | bc)
max_train_samples=$(echo "${num_tokens} / $block_size" | bc)
global_bs=$(echo "$per_device_train_batch_size * $gradient_accumulation_steps * $num_nodes * $num_gpu_per_node" | bc)
tokens_per_batch=$(echo "$global_bs * $block_size" | bc)

echo "=== 训练参数 ==="
echo "max_steps: $max_steps"
echo "global batch size: $global_bs"
echo "#tokens/batch: $tokens_per_batch"

# 输出目录设置（使用时间戳替代SLURM_JOBID）
timestamp=$(date +%Y%m%d-%H%M%S)
output_dir="outputs/${job_name}-${timestamp}"
mkdir -p $output_dir
echo "输出目录: $output_dir"

# 保存配置信息
git diff > $output_dir/diff.patch
env > $output_dir/.env
echo $comment > $output_dir/comment.txt

# ========== 单节点配置（注释多节点设置） ==========
# ##############################################################
# ### 重要：单节点无需网络配置 ###
# # master_addr="100.64.1.25"          # 已注释
# # master_port=29518                  # 已注释
# # node_rank=0                        # 已注释
# ##############################################################

# ========== 修改后的训练命令 ==========
deepspeed \
  --num_gpus 8 \
  smoe/entrypoint/cpt/cpt_fpt.py  --deepspeed ${deepspeed_config_file} \
  --model_name_or_path ${pretrained_model} \
  --model_type ${model_type} \
  --tokenizer_name_or_path ${tokenizer_path} \
  --dataset_dir ${dataset_dir} \
  --data_cache_dir resources/cache \
  --validation_dir ${validation_dir} \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${per_device_eval_batch_size} \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --seed ${seed} \
  --bf16 \
  --num_train_epochs 1 \
  --final_lr_portion ${final_lr_portion} \
  --optim adamw_torch \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --learning_rate ${lr} \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --warmup_steps 8000 \
  --max_steps ${max_steps} \
  --max_train_samples ${max_train_samples} \
  --save_strategy steps \
  --save_total_limit 1 \
  --save_steps 1000 \
  --dataloader_num_workers 0 \
  --dataloader_pin_memory True \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --block_size ${block_size} \
  --output_dir ${output_dir} \
  --overwrite_output_dir \
  --ddp_timeout 30000 \
  --ddp_find_unused_parameters False \
  --torch_dtype bfloat16 \
  --gradient_checkpointing \
  --logging_first_step True \
  --logging_strategy steps \
  --logging_steps 10 \
  --log_level info \
  --log_level_replica warning \
  --log_on_each_node False \
  --report_to none \
  --gate_type "TopKBalancedNoisyGate" \
  --calculator_type "UniversalCalculator" \
  --num_selects ${num_selects}
