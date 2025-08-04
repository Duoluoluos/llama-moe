import math
import pathlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import transformers
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from smoe.utils.conversation import Conversation
from smoe.utils.io import load_jsonlines
from llamafactory.train import LLaMATrainer, LLaMATrainingArguments
from llamafactory.model import load_model_and_tokenizer

IGNORE_TOKEN_ID = -100


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained tokenizer"}
    )
    trust_remote_code: bool = field(
        default=True, metadata={"help": "Whether or not to allow for custom models defined on the Hub"}
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    model_type: str = field(
        default="auto", metadata={"help": "Model type: `moe` or `mixtral` or `auto`"}
    )
    torch_dtype: str = field(
        default="auto", metadata={"help": "Torch dtype: `float32` or `bfloat16`"}
    )
    additional_config: str = field(
        default=None, metadata={"help": "Additional config file (in json) to load"}
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={"help": "attention implementation, choice from [eager, flash_attention_2, sdpa]"}
    )

    def __post_init__(self):
        if hasattr(torch, self.torch_dtype):
            self.torch_dtype = getattr(torch, self.torch_dtype)
        if self.additional_config is not None:
            if not pathlib.Path(self.additional_config).exists():
                raise ValueError(f"Additional config file {self.additional_config} not found")
            from smoe.utils.io import load_json
            self.additional_config = load_json(self.additional_config)


@dataclass
class DataArguments:
    eval_data_dir: str = field(
        default=None, metadata={"help": "Path to the evaluation data folder."}
    )
    dataset_dir_or_path: str = field(
        default="data/merged", metadata={"help": "Path to dataset directory or a single jsonl file"}
    )


@dataclass
class TrainingArguments(LLaMATrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    train_only_gate: bool = field(
        default=False,
        metadata={"help": "Whether to only train the gate during training."},
    )
    save_final_ckpt: bool = field(
        default=True,
        metadata={"help": "Whether to save final checkpoint."},
    )


class CachedJsonlDataset(Dataset):
    def __init__(
        self,
        datapath: str,
        tokenizer: PreTrainedTokenizer,
        seed: int = 1227,
    ) -> None:
        super().__init__()
        self.datapath = datapath
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.data = load_jsonlines(datapath)
        self.rng.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        ins = self.data[index]
        processed = preprocess([ins], self.tokenizer)
        ins = {}
        for key in processed:
            ins[key] = processed[key][0]
        return ins

    def state_dict(self):
        return {
            "datapath": self.datapath,
            "seed": self.seed,
            "rng": self.rng.getstate(),
        }


def preprocess(
    instances,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    tokenizer_legacy = getattr(tokenizer, "legacy", True)
    conv = Conversation()
    conv.sep2 = tokenizer.eos_token
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, ins in enumerate(instances):
        if roles[ins["conversations"][0]["from"]] != roles["human"]:
            # Skip the first one if it is not from human
            ins["conversations"] = ins["conversations"][1:]

        conv.clear_msg()
        sys_msg = ins.get("system_prompt")
        if sys_msg is not None:
            conv.set_system_message(sys_msg)
        else:
            conv.set_system_message("")
        for j, turn in enumerate(ins["conversations"]):
            role = roles[turn["from"]]
            assert role == conv.roles[j % 2], f"{i}/{j}"
            conv.append_message(role, turn["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    res = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    input_ids = res["input_ids"]
    attention_masks = res["attention_mask"]
    targets = input_ids.clone()

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target, attention_mask in zip(
        conversations, targets, attention_masks
    ):
        turns = conversation.split(conv.sep2)
        total_len = attention_mask.sum()

        cur_len = 0
        has_bos = False
        if target[0] == tokenizer.bos_token_id:
            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID  # bos token
            has_bos = True
        for i, turn in enumerate(turns):
            if turn == "":
                break
            # +1: add sep2 token
            turn_len = len(tokenizer(turn).input_ids) - int(has_bos) + 1

            # sep: " ASSISTANT: "
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct: bos and the last space token
            # -1 means remove extra suffix space in sep
            instruction_len = len(tokenizer(parts[0]).input_ids) - int(has_bos) - 1

            if i != 0 and not tokenizer_legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

            if i != 0 and not tokenizer_legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                logger.info(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_masks,
    )


def fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    if not isinstance(features[0], Dict):
        try:
            features = [vars(f) for f in features]
        except TypeError:
            print(len(features), type(features[0]), features[0])
    first = features[0]
    batch = {}

    # Special handling for labels.
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    try:
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
    except ValueError:  # quick fix by simply take the first example
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([features[0][k]] * len(features))
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([features[0][k]] * len(features)))
                else:
                    batch[k] = torch.tensor([features[0][k]] * len(features))

    return batch


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    # Load model and tokenizer using llamafactory
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_args.model_name_or_path,
        tokenizer_name_or_path=model_args.tokenizer_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side=model_args.padding_side,
        torch_dtype=model_args.torch_dtype,
        additional_config=model_args.additional_config,
        attn_impl=model_args.attn_impl,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )

    # Freeze all parameters except gate if train_only_gate is True
    if training_args.train_only_gate:
        for name, param in model.named_parameters():
            if "gate" not in name:
                param.requires_grad = False
        logger.info("Only gate parameters are trainable.")

    # Prepare dataset
    train_dataset = None
    datapath = pathlib.Path(data_args.dataset_dir_or_path)
    if not datapath.exists():
        raise ValueError(f"Dataset path {datapath} not found")
    elif datapath.is_file():
        logger.info(f"CachedJsonlDataset: {datapath}")
        train_dataset = CachedJsonlDataset(
            data_args.dataset_dir_or_path,
            tokenizer,
            seed=training_args.seed,
        )
    else:
        raise ValueError(f"Unknown dataset path type: {datapath}")
    logger.info("train dataset ready")

    # Initialize trainer using llamafactory's LLaMATrainer
    trainer = LLaMATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=fault_tolerance_data_collator,
    )
    logger.info("trainer ready")

    # Start training
    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            logger.info("resume training from ckpt")
            trainer.train(resume_from_checkpoint=True)
        else:
            logger.info("start training")
            trainer.train()

    # Save model
    if training_args.save_final_ckpt:
        logger.info("training finished, dumping model")
        model.config.use_cache = True
        trainer.save_state()
        trainer.save_model()

    logger.info("🎉 All done~")


if __name__ == "__main__":
    train()