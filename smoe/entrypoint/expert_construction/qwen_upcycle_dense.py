import argparse
import os

from smoe.utils.expert_construction.upcycle_dense import upcycle_qwen_models_to_moe, align_all_layers
from smoe.utils.operations.operation_string import str2bool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple Llama models into a single Mixture-of-Experts (MoE) model.")
    parser.add_argument('--template_path',
                        type=str,
                        default=None,
                        help='')
    parser.add_argument('--model_paths',
                        nargs='+',  # 关键改动：允许接受一个或多个模型路径
                        required=True,
                        help='A list of paths to the component Llama models to be merged into experts.')
    parser.add_argument('--select_file_path',
                        type=str,
                        default=None,
                        help='Path to the directory containing pretrained gate weight files. Required if use_random_gate is False.')
    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help='Path to save the merged LlamaMoE model.')
    parser.add_argument('--template',
                        type=str,
                        default='layers.{}.mlp.gate_proj.weight',
                        help='Template for loading gate weight files.')

    # --- MoE 配置参数 ---
    parser.add_argument('--num_selects',
                        type=int,
                        default=2,
                        help='Number of experts to select for each token.')

    # --- 门控网络配置 ---
    parser.add_argument('--use_random_gate',
                        type=str,
                        default="False",
                        help='Use randomly initialized gate weights instead of loading them.')
    parser.add_argument('--use_softmax',
                        type=str,
                        default='True',
                        help='Whether to use softmax for gating scores.')
    parser.add_argument('--multiply_gate_scores',
                        type=str,
                        default='True',
                        help='Whether to multiply gate scores.')
    parser.add_argument('--score_scale_factor',
                        type=float,
                        default=1.0,
                        help='Scale factor for gating scores.')
    parser.add_argument('--stage',
                        type=str,
                        default='') 
    parser.add_argument('--codata_path',
                        type=str,
                        default='') 
    args = parser.parse_args()

    # --- 参数处理 ---
    args.use_random_gate = str2bool(args.use_random_gate)
    args.use_softmax = str2bool(args.use_softmax)
    args.multiply_gate_scores = str2bool(args.multiply_gate_scores)

    if not args.use_random_gate and not args.select_file_path:
        raise ValueError("The '--select_file_path' argument is required when 'use_random_gate' is False.")

    print("--- Merge Configuration ---")
    print(args, "\n")

    # --- 调用核心合并函数 ---
    # 我们只有一个合并函数，所以不需要 convert_type 判断
    if args.stage == "merge":
        upcycle_qwen_models_to_moe(
            template_path=args.template_path,
            model_paths=args.model_paths,
            save_path=args.save_path,
            num_selects=args.num_selects,
            score_scale_factor=args.score_scale_factor,
            use_random_gate=args.use_random_gate,
            use_softmax=args.use_softmax,
            multiply_gate_scores=args.multiply_gate_scores,
        )
    elif args.stage == 'alignment':
        align_all_layers(
            template_path=args.template_path,
            jsonl_path=args.codata_path,
            save_path=args.save_path
        )
    else:
        raise ValueError("Please specify stage using --stage flag.")
    print("\nDone.")