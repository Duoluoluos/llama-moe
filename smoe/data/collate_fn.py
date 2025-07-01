from typing import Any, Mapping
from typing import List, Dict
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils



def collate_fn_lm(
    batch: List[Dict],
    pad_token_id: int = 0,
    block_size: int | None = 128,
    return_labels: bool = True,
):
    """
    通用 Language-Modeling collate_fn.
    
    Args
    ----
    batch : list[dict]            # DataLoader 传进来的一个 batch
    pad_token_id : int            # tokenizer.pad_token_id；若没有 pad_token，可用 0
    block_size : int | None       # 若传入则强制截断/填充到固定长度
    return_labels : bool          # 若为 True，则把 labels = input_ids.clone()
    
    Returns
    -------
    dict[str, torch.Tensor]       # keys: input_ids, attention_mask, (labels)
    """
    # print(batch)
    # 1) 原始序列
    seqs = [example["input_ids"] for example in batch]
    lengths = [len(s) for s in seqs]

    # 2) 计算目标长度
    if block_size is None:
        max_len = max(lengths)                      # 动态 pad
    else:
        max_len = block_size                        # 固定长度（比如 2048）

    # 3) 预分配张量（B, L）
    B = len(seqs)
    input_ids = torch.full((B, max_len),
                           pad_token_id,
                           dtype=torch.long)
    attention_mask = torch.zeros((B, max_len),
                                 dtype=torch.long)

    # 4) 填充
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]                         # 截断（若太长）
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.as_tensor(seq, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

    batch_dict = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
    }
    if return_labels:
        batch_dict["labels"] = input_ids.clone()

    return batch_dict




def fault_tolerance_data_collator(features: list) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
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
    # Again, we will use the first element to figure out which key/values are not None for this model.

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


def identity_collator(examples):  # 不对数据进行处理
    return examples


def tensor_cat_collator(examples):  # 拼接tensor
    return torch.cat(examples, dim=0)


class tensor_cat_padding_collater:  # 拼接tensor，并padding到最大长度
    def __init__(self, padding_id, padding_position="right", return_padding_mask=True):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

    def __call__(self, examples):
        if self.padding_position == "right":
            padded_examples = rnn_utils.pad_sequence(
                examples, batch_first=True, padding_value=self.padding_id
            )
        elif (
            self.padding_position == "left"
        ):  # This will take about twice the time compared to right padding
            flipped_examples = [torch.flip(tensor, dims=[0]) for tensor in examples]
            padded_examples_flip = rnn_utils.pad_sequence(
                flipped_examples, batch_first=True, padding_value=self.padding_id
            )
            padded_examples = torch.flip(padded_examples_flip, dims=[1])
        else:
            raise NotImplementedError

        if self.return_padding_mask:
            padding_mask = padded_examples != self.padding_id
            return padded_examples, padding_mask
        else:
            return padded_examples


def tensor_list_cat_collator(examples):  # 拼接list中对应位置的tensor，返回list
    return [
        torch.cat([tensor[i] for tensor in examples], dim=0)
        for i in range(len(examples[0]))
    ]


class tensor_list_cat_padding_collater:  # 拼接list中对应位置的tensor，并padding到最大长度，返回list
    def __init__(self, padding_id, padding_position="right", return_padding_mask=True):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

    def __call__(self, examples):
        num_tensors = len(examples[0])
        padded_tensors = []
        padding_masks = []

        for i in range(num_tensors):
            tensor_list = [example[i] for example in examples]

            if self.padding_position == "right":
                padded_tensor = rnn_utils.pad_sequence(
                    tensor_list, batch_first=True, padding_value=self.padding_id
                )
            elif (
                self.padding_position == "left"
            ):  # This will take about twice the time compared to right padding
                flipped_tensors = [
                    torch.flip(tensor, dims=[0]) for tensor in tensor_list
                ]
                padded_tensors_flip = rnn_utils.pad_sequence(
                    flipped_tensors, batch_first=True, padding_value=self.padding_id
                )
                padded_tensor = torch.flip(padded_tensors_flip, dims=[1])
            else:
                raise NotImplementedError

            padded_tensors.append(padded_tensor)
            if self.return_padding_mask:
                padding_masks.append(padded_tensors[i] != self.padding_id)

        if self.return_padding_mask:
            return padded_tensors, padding_masks
        else:
            return padded_tensors


def tensor_dict_cat_collator(examples):  # 拼接dict中对应位置的tensor，返回dict
    return {
        key: torch.cat([example[key] for example in examples], dim=0)
        for key in examples[0].keys()
    }
