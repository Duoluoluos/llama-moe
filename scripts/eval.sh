export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_ALLOW_CODEW_EVAL=1
lm_eval --model hf \
    --model_args pretrained=/hdd-cifs/wangqi/meld_models/LoRA_Editted/Qwen2.5-MetaMathQA \
    --tasks mmlu \
    --device cuda:2 \
    --batch_size 64 \
    --output_path logs/M2_mmlu.jsonl

lm_eval --model hf \
    --model_args pretrained=/hdd-cifs/wangqi/meld_models/LoRA_Editted/Qwen2-SciQA \
    --tasks mmlu \
    --device cuda:4 \
    --batch_size 64 \
    --output_path logs/M4_mmlu.jsonl