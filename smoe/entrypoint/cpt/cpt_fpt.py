import logging
import os
import socket
import sys
from pathlib import Path
import torch.distributed as dist
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import datasets
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    LlamaTokenizer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset as HFDataset

sys.path.append('.')
from smoe.callbacks.save_model import SchedulerStateCallback
from smoe.callbacks.tensorboard import EnhancedTensorboardCallback
from smoe.data.collate_fn import fault_tolerance_data_collator, collate_fn_lm
from smoe.data.dynamic_selection import (
    AVERAGE_SLIMPAJAMA_DATA_PORTION,
    LLAMA_DATA_PORTION,
    SHEAREDLLAMA_DATA_PORTION,
    TOY_DATA
)
from smoe.data.streaming import CachedJsonlDataset, SubDirWeightedPackedJsonlDataset, PackedJsonlDataset, load_process_and_create_hf_dataset
from smoe.metrics.preprocess import logits_argmax
from smoe.models.llama_moe.configuration_llama_moe import LlamaMoEConfig
from smoe.models.llama_moe.modeling_llama_moe import LlamaMoEForCausalLM
from smoe.models.llama_moe_residual import (
    LlamaMoEResidualConfig,
    LlamaMoEResidualForCausalLM,
)
from smoe.models.mixtral.configuration_mixtral import MixtralConfig
from smoe.models.mixtral.modeling_mixtral import MixtralForCausalLM
# from smoe.modules.flash_attn import replace_xformers
from smoe.trainer.llama_lr_scheduling import LlamaLrSchedulingTrainer
from smoe.utils.config import (
    DataArguments,
    EnhancedTrainingArguments,
    ModelArguments,
    parse_args,
)
from smoe.utils.notification import wechat_sender
from smoe.utils.param import get_trainable_parameters
from smoe.utils.debugging import cast_all_buffers
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


MODEL_MAP = {
    "qwen2": Qwen2ForCausalLM,
    "llama": LlamaForCausalLM,
    "llama_moe": LlamaMoEForCausalLM,
    "llama_moe_residual": LlamaMoEResidualForCausalLM,
}

CONFIG_MAPPING.update(
    {
        "llama": LlamaConfig,
        "llama_moe": LlamaMoEConfig,
        "llama_moe_residual": LlamaMoEResidualConfig,
    }
)


logger = logging.getLogger(__name__)


# @wechat_sender(msg_prefix="CPT Training")
def main():
    dist.init_process_group(
        backend="nccl",  
        init_method="env://",  
        world_size=int(os.environ.get("WORLD_SIZE", 1)),  
        rank=int(os.environ.get("RANK", 0))  
    )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    model_args, data_args, training_args = parse_args(
        ModelArguments, DataArguments, EnhancedTrainingArguments
    )
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    hostname = socket.gethostname()
    logger.warning(
        f"Global rank: {training_args.process_index}, "
        f"Host: {hostname}, IP: {socket.gethostbyname(hostname)}, "
        f"Process local rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"fp16 training: {training_args.fp16}, "
        f"bf16 training: {training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is"
                " not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid"
                " this behavior, change the `--output_dir` or add"
                " `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    logger.info(f"Seed set to: {training_args.seed}")
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "gate_type": model_args.gate_type,
        "calculator_type": model_args.calculator_type,
        "num_selects": model_args.num_selects,
        "gate_network": model_args.gate_network_type,
        "score_scale_factor": model_args.moe_calculator_score_scale_factor,
        "gate_balance_loss_weight": model_args.gate_balance_loss_weight,
    }
    ConfigClass = AutoConfig
    if model_args.config_name == "llama_moe" or model_args.model_type == "llama_moe":
        ConfigClass = LlamaMoEConfig
    elif (
        model_args.config_name == "llama_moe_residual"
        or model_args.model_type == "llama_moe_residual"
    ):
        ConfigClass = LlamaMoEResidualConfig

    if model_args.config_name:
        config = ConfigClass.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = ConfigClass.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    if training_args.gradient_checkpointing:
        config.use_cache = False


    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "legacy": True if model_args.use_legacy_tokenizer else False,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.tokenizer_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path, use_fast=True)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported"
            " by this script.You can do it from another script, save it, and load it"
            " from here, using --tokenizer_name."
        )
    
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than"
                " the default `block_size` value of 1024. If you would like to use a"
                " longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the"
                f" maximum length for the model({tokenizer.model_max_length}). Using"
                f" block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    prob_map = TOY_DATA


    with training_args.main_process_first(desc="dataset map tokenization and grouping"):
        lm_datasets = PackedJsonlDataset(
            data_args.dataset_dir,
            seed=training_args.seed,
            block_size=data_args.block_size,
        )
    if training_args.do_train:
        train_dataset = lm_datasets
        if data_args.max_train_samples is None:
            raise ValueError("max_train_samples cannot be None")
        # logger.info("training example:")
        res = None
        if hasattr(train_dataset, "take"):
            res = tokenizer.decode([x["input_ids"] for x in train_dataset.take(1)][0])
        else:
            for x in train_dataset:
                #logger.info(x)
                input_ids = x["input_ids"]
                break
            res = tokenizer.decode(input_ids)
        #logger.info(f"example res:{res}")

    eval_dataset = None
    if training_args.do_eval:
        validation_dir = Path(data_args.validation_dir)
        jsonl_files = list(validation_dir.glob("*.jsonl"))
        if len(jsonl_files) == 0:
            raise FileNotFoundError(f"Evaluation is enabled, but no .jsonl files were found in {validation_dir}")
        
        if len(jsonl_files) > 1:
            raise ValueError(f"Multiple .jsonl files found in {validation_dir}. Please specify a directory with only one validation file.")

        validation_file_path = jsonl_files[0]
        eval_dataset = load_process_and_create_hf_dataset(data_args.block_size, validation_file_path, debug = False)


    if training_args.do_predict:
        predict_dataset = lm_datasets
        if data_args.max_predict_samples is None:
            raise ValueError("max_predict_samples cannot be None")

    if training_args.do_train:
        torch_dtype = ( model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
        )
    # zhutong: this is for debug usage only
    if training_args.debug_mode:
        debug_config = config.to_dict()
        # 用户原有config参数基础上覆盖以下值
        debug_config = {
            "hidden_size": 512,  # 原4096
            "intermediate_size": 1024,  # 原11008
            "num_hidden_layers": 2,  # 原32
            "num_attention_heads": 4,  # 原32
            "num_key_value_heads": 2,  # 保持与attention heads比例
            "num_experts": 4,  # 原16
            "num_selects": 2,  # 原4
            "gate_add_noise": False,  # 关闭噪声
            "gate_balance_loss_weight": 0,  # 关闭负载均衡损失
            "capacity_factor": 1.0,  # 关闭buffer
            "use_cache": False,  # 关闭KV缓存
            "vocab_size": tokenizer.vocab_size,
        }
        debug_config = config.__class__(**debug_config)
        logger.warning(f"DEBUG MODE: Creating minimal model:{debug_config}")
        with init_empty_weights():
            model = LlamaMoEForCausalLM(debug_config)
            model = model.to_empty(device=device) 
            
    else:
        # Preprocessing the datasets.
        if 'moe' in model_args.model_type:

            ModelClass = MODEL_MAP[model_args.model_type]

            with init_empty_weights():
                model = ModelClass(config)  
            
            model = load_checkpoint_and_dispatch(
                model,
                checkpoint=model_args.model_name_or_path,
                device_map={"": device.index},          # 每个 rank 放到自己那张卡
                dtype=torch_dtype,
                no_split_module_classes=[
                    "LlamaDecoderLayer", "LlamaMoEDecoderLayer"
                ],
            )

        else:
            model = AutoModelForCausalLM.from_config(config)
            n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
            logger.info(
                f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params"
            )
        model.to(device)
        logger.info(f'Params:{model.named_parameters()}')

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


        model_vocab_size = model.get_output_embeddings().weight.size(0)

    # Set
    model = DDP( model,
    device_ids=[local_rank],        # 也可以直接省略这两个参数
    output_device=local_rank,       # （若省略则让 DDP 自动推断）
    find_unused_parameters=False
    )

    trainable_params, _ = get_trainable_parameters(model, verbose=True)
    training_args.num_training_params = trainable_params

    # Initialize our Trainer
    trainer = LlamaLrSchedulingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=fault_tolerance_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=(
            logits_argmax
            if training_args.do_eval
            else None
        ),
        model_type=model_args.model_type,
    )
    trainer.add_callback(EnhancedTensorboardCallback)
    trainer.add_callback(SchedulerStateCallback)
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = data_args.max_train_samples

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(ignore_keys=None)
        logger.info(f"{metrics}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()