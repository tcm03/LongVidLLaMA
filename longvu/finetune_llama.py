# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.
import inspect
from functools import partial
import copy
import datetime
import json
import os
import pathlib
import uuid
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Optional, Sequence
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

import torch

import transformers

from decord import cpu, VideoReader

import sys
sys.path.append('.')
from longvu import conversation as conversation_lib

from longvu.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from longvu.language_model.cambrian_llama import CambrianLlamaForCausalLM
from longvu.language_model.cambrian_qwen import CambrianQwenForCausalLM
from longvu.mm_datautils import (
    preprocess,
    preprocess_multimodal,
    safe_save_model_for_hf_trainer,
    smart_tokenizer_and_embedding_resize,
)
from longvu.mm_trainer import LLaVATrainer
from PIL import Image, ImageSequence

from tensorboard.compat.tensorflow_stub.io.gfile import register_filesystem
from torch import distributed as dist

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
from mm_utils import extract_engagement_label
from transformers.utils import logging as hf_logging  # Hugging Face logger

from transformers.integrations import TensorBoardCallback
TENSORBOARD_LOG_DIR_NAME: str = "tensorboard_logs"

##### LOGGING CONFIGURATION ####

# Ensure logs directory exists
log_dir = "runtime_logs"
os.makedirs(log_dir, exist_ok=True)
# Generate a unique log filename using timestamp
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(log_dir, f"run_{timestamp}.log")

import logging
# Create a logger
tcm_logger = logging.getLogger("tcm_logger")
tcm_logger.setLevel(logging.DEBUG)  # Capture all logs

# Create a file handler for logging
file_handler = logging.FileHandler(log_filename, mode="w")
file_handler.setLevel(logging.DEBUG)  # Capture all log levels

# Define log format
formatter = logging.Formatter(
    "%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)

# Add the file handler to the logger
tcm_logger.addHandler(file_handler)

# Prevent log propagation to the root logger
tcm_logger.propagate = False

#hf_logger = hf_logging.get_logger("transformers")
#hf_logger.setLevel(logging.INFO)  # Ensure it logs debug messages
#
## Attach the same file handler to Hugging Face's logger
#hf_logger.addHandler(file_handler)
#hf_logger.propagate = False  # Prevent duplicate logs

##### DONE LOGGING CONFIGURATION ####
from longvu.resource_logging import *

@dataclass
class ModelArguments:
    input_model_filename: Optional[str] = field(default=None)
    output_model_filename: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    grid_size: Optional[int] = field(default=8)
    vision_tower_type: Optional[str] = field(default="sam")
    mm_hidden_size: Optional[int] = field(default=256)

    # cambrian
    vision_tower_aux_list: Optional[str] = field(
        default='["siglip/CLIP-ViT-SO400M-14-384", "facebook/dinov2-giant-res378"]'
    )
    vision_tower_aux_token_len_list: Optional[str] = field(default="[576, 576]")
    image_token_len: Optional[int] = field(default=576)
    num_query_group: Optional[int] = field(default=1)
    query_num_list: Optional[str] = field(default="[576]")
    connector_depth: Optional[int] = field(default=3)
    vision_hidden_size: Optional[int] = field(default=1024)
    connector_only: bool = field(default=True)
    num_of_vision_sampler_layers: Optional[int] = field(default=10)
    start_of_vision_sampler_layers: Optional[int] = field(default=0)
    stride_of_vision_sampler_layers: Optional[int] = field(default=3)

    is_st_sampler: bool = field(default=False)
    highres_connect: bool = field(default=False)
    highres: bool = field(default=False)
    connect_layer: Optional[int] = field(default=2)
    lowres_token: Optional[int] = field(default=8)
    dino_threshold: float = field(default=0.83)
    drop_threshold: float = field(default=0.8)
    frame_pos: bool = field(default=False)
    is_image_newline: bool = field(default=True)


@dataclass
class DataArguments:
    data_path_train: Optional[str] = field(default=None)
    data_path_val: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_position: Optional[int] = field(default=91)
    image_folder: Optional[str] = field(default=None)
    uniform_sample: bool = field(default=False)
    image_aspect_ratio: str = "square"
    num_points: int = field(default=0)
    video_fps: float = field(default=1)
    use_subtitle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_text_decoder: bool = field(default=False)
    unfreeze_mm_vision_tower: bool = field(default=False)
    mm_vision_tower_lr: Optional[float] = None
    unfreeze_mm_image_decoder: bool = field(default=False)
    # deepspeed: Optional[str] = field(default="ds_config.json")
    # is_deepspeed_zero3_enabled: bool = True

    mm_vision_sampler_lr: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    model_max_length: Optional[int] = field(default=8192)

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # For our seq2seq task of explaining engagement prediction, we need inputs in compute_metrics()
    # Discuss: https://discuss.huggingface.co/t/how-to-accessing-the-input-ids-in-evalprediction-predictions-in-seq2seqtrainer/25372
    include_inputs_for_metrics: Optional[bool] = field(default=False)
    torch_empty_cache_steps: Optional[int] = field(default=None)
    eval_accumulation_steps: Optional[int] = field(default=2)
    # @tcm: Somehow I use activation_checkpointing and OOM right in the training phase! with gradient_checkpointing_kwargs and non-reentrant, still OOM during training (step 46 on snapugc_mini)
    # activation_checkpointing: bool = field(default=True)
    # gradient_checkpointing_kwargs={'use_reentrant': False}


def get_local_rank() -> int:
    if os.environ.get("LOCAL_RANK"):
        return int(os.environ["LOCAL_RANK"])
    else:
        return torch.distributed.get_rank()


def get_global_rank() -> int:
    """
    Get rank using torch.distributed if available. Otherwise, the RANK env var instead if initialized.
    Returns 0 if neither condition is met.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    environ_rank = os.environ.get("RANK", "")
    if environ_rank.isdecimal():
        return int(os.environ["RANK"])

    return 0


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def get_padding_offset(cur_size, original_size):
    cur_w, cur_h = cur_size
    original_w, original_h = original_size

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        return 0, 0, padding, padding
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        return padding, padding, 0, 0


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def prepare_image_info(image_size, image_token_len, newline=False):
    num_tokens_per_side = int(image_token_len**0.5)
    if newline:
        # for the newline embedding
        attention_mask = torch.ones(
            num_tokens_per_side, num_tokens_per_side + 1, dtype=torch.bool
        )
    else:
        attention_mask = torch.ones(
            num_tokens_per_side, num_tokens_per_side, dtype=torch.bool
        )
    left_offset, right_offset, top_offset, bottom_offset = get_padding_offset(
        (num_tokens_per_side, num_tokens_per_side), image_size
    )
    if newline:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset - 1 : -1] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :] = 0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    else:
        if left_offset > 0:
            attention_mask[:, :left_offset] = 0
        if right_offset > 0:
            attention_mask[:, -right_offset:] = 0
        if top_offset > 0:
            attention_mask[:top_offset, :] = 0
        if bottom_offset > 0:
            attention_mask[-bottom_offset:, :] = 0
    attention_mask = attention_mask.flatten()
    position_ids = attention_mask.cumsum(0) - 1
    return attention_mask, position_ids


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def prepare_multimodal_data(
    input_ids,  # pyre-fixme[2]
    labels,  # pyre-fixme[2]
    attention_mask,  # pyre-fixme[2]
    image_sizes,  # pyre-fixme[2]
    image_token_len=576,  # pyre-fixme[2]
    image_aux_token_len_list=[192 * 192],  # pyre-fixme[2]
    max_length=2048,  # pyre-fixme[2]
):
    input_ids_im_replaced = []
    labels_im_replaced = []
    attention_mask_im_replaced = []
    position_ids_im_replaced = []
    im_aux_attention_masks_list = [[] for _ in range(len(image_aux_token_len_list))]
    base_image_token_len_per_side = int(image_token_len**0.5)
    image_aux_token_len_per_side_list = [
        int(image_aux_token_len_per_side**0.5)
        for image_aux_token_len_per_side in image_aux_token_len_list
    ]
    # insert the padding tokens to the places of image so we can embed them together
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        assert num_images == 1, num_images
        image_size = image_sizes[batch_idx]

        image_token_indices = (
            [-1]
            + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            + [cur_input_ids.shape[0]]
        )

        cur_input_ids_im_replaced = []
        cur_labels_im_replaced = []
        cur_attention_mask_im_replaced = []
        cur_position_ids_im_replaced = []

        cur_labels = labels[batch_idx]
        cur_attention_mask = attention_mask[batch_idx]
        index = 0
        for i in range(len(image_token_indices) - 1):
            # still keep the first image token in input_ids for further use
            cur_input_ids_im_replaced.append(
                cur_input_ids[
                    image_token_indices[i] + 1 : image_token_indices[i + 1] + 1
                ]
            )
            cur_labels_im_replaced.append(
                cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
            )
            cur_attention_mask_im_replaced.append(
                cur_attention_mask[
                    image_token_indices[i] + 1 : image_token_indices[i + 1]
                ]
            )
            cur_position_ids_im_replaced.append(
                torch.arange(
                    index,
                    index + image_token_indices[i + 1] - (image_token_indices[i] + 1),
                    dtype=torch.long,
                    device=cur_input_ids.device,
                )
            )
            index += image_token_indices[i + 1] - (image_token_indices[i] + 1)

            if i < len(image_token_indices) - 2:
                num_tokens_per_side = int(image_token_len**0.5)
                image_token_len_with_newline = image_token_len + num_tokens_per_side
                cur_input_ids_im_replaced.append(
                    torch.full(
                        (image_token_len_with_newline - 1,),
                        0,
                        device=cur_input_ids.device,
                        dtype=cur_input_ids.dtype,
                    )
                )
                cur_labels_im_replaced.append(
                    torch.full(
                        (image_token_len_with_newline,),
                        IGNORE_INDEX,
                        device=cur_labels.device,
                        dtype=cur_labels.dtype,
                    )
                )

                cur_im_attention_mask, cur_im_position_ids = prepare_image_info(
                    image_size, image_token_len, newline=True
                )

                for aux_i, image_aux_token_len_per_side in enumerate(
                    image_aux_token_len_per_side_list
                ):
                    assert image_aux_token_len_per_side >= base_image_token_len_per_side
                    num_base_crops_per_aux_side = (
                        image_aux_token_len_per_side // base_image_token_len_per_side
                    )

                    cur_im_aux_attention_mask, _ = prepare_image_info(
                        image_size, image_aux_token_len_per_side**2
                    )
                    cur_im_aux_attention_mask = cur_im_aux_attention_mask.view(
                        base_image_token_len_per_side,
                        num_base_crops_per_aux_side,
                        base_image_token_len_per_side,
                        num_base_crops_per_aux_side,
                    )
                    cur_im_aux_attention_mask = (
                        cur_im_aux_attention_mask.permute(0, 2, 1, 3)
                        .contiguous()
                        .flatten(0, 1)
                        .flatten(1, 2)
                    )
                    cur_im_aux_attention_mask[
                        cur_im_aux_attention_mask.sum(dim=1) == 0
                    ] = True
                    im_aux_attention_masks_list[aux_i].append(cur_im_aux_attention_mask)
                cur_im_position_ids += index

                if cur_attention_mask[image_token_indices[i + 1]]:
                    cur_attention_mask_im_replaced.append(cur_im_attention_mask)
                    cur_position_ids_im_replaced.append(
                        cur_im_position_ids.to(torch.long)
                    )
                    index = cur_im_position_ids.max() + 1
                else:
                    num_tokens_per_side = int(image_token_len**0.5)
                    image_token_len_with_newline = image_token_len + num_tokens_per_side
                    cur_attention_mask_im_replaced.append(
                        torch.full(
                            (image_token_len_with_newline,),
                            0,
                            device=cur_attention_mask.device,
                            dtype=cur_attention_mask.dtype,
                        )
                    )
                    cur_position_ids_im_replaced.append(
                        torch.full(
                            (image_token_len_with_newline,),
                            0,
                            device=cur_input_ids.device,
                            dtype=torch.long,
                        )
                    )

        input_ids_im_replaced.append(torch.cat(cur_input_ids_im_replaced))
        labels_im_replaced.append(torch.cat(cur_labels_im_replaced))
        attention_mask_im_replaced.append(torch.cat(cur_attention_mask_im_replaced))
        position_ids_im_replaced.append(torch.cat(cur_position_ids_im_replaced))

    # Truncate sequences to max length as image embeddings can make the sequence longer
    new_input_ids = [x[0:max_length] for x in input_ids_im_replaced]
    new_labels = [x[0:max_length] for x in labels_im_replaced]
    new_attention_mask = [x[0:max_length] for x in attention_mask_im_replaced]
    new_position_ids = [x[0:max_length] for x in position_ids_im_replaced]
    new_input_ids = torch.stack(new_input_ids)
    new_labels = torch.stack(new_labels)
    new_attention_mask = torch.stack(new_attention_mask)
    new_position_ids = torch.stack(new_position_ids)
    im_aux_attention_masks_list = [
        torch.stack(im_aux_attention_masks)
        for im_aux_attention_masks in im_aux_attention_masks_list
    ]
    return (
        new_input_ids,
        new_labels,
        new_attention_mask,
        new_position_ids,
        im_aux_attention_masks_list,
    )

class MetricsAccumulator:

    def __init__(self):
        self.golds = []
        self.preds = []

    def update(self, golds, preds):
        assert len(golds) == len(preds), "require equal labels and prediction arrays"
        self.golds.extend(golds)
        self.preds.extend(preds)

    def reset(self):
        self.golds = []
        self.preds = []

    def length(self):
        return len(self.golds)

    def compute(self):
        accuracy = accuracy_score(self.golds, self.preds)
        prec_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            self.golds, self.preds, average="weighted"
        )
        prec_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            self.golds, self.preds, average="micro"
        )
        prec_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            self.golds, self.preds, average="macro"
        )
        return {
            "accuracy": accuracy,
            "precision_weighted": prec_w,
            "recall_weighted": recall_w,
            "f1_weighted": f1_w,
            "precision_micro": prec_micro,
            "recall_micro": recall_micro,
            "f1_micro": f1_micro,
            "precision_macro": prec_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }

global_metrics_accumulator = MetricsAccumulator()

# def compute_metrics(eval_pred, tokenizer):
def compute_metrics(eval_pred, tokenizer, compute_result):
    """
    Computes accuracy, precision, recall, and F1-score for the sequence classification task.

    Args:
    eval_pred (EvalPrediction): An object containing predictions and references (true labels).

    Returns:
    dict: A dictionary with metric names as keys and their values.
    """

    preds = eval_pred.predictions
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    labels = eval_pred.label_ids
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    inputs = eval_pred.inputs
    if isinstance(inputs, dict):
        inputs = inputs["input_ids"]
    masks = eval_pred.masks
    tcm_logger.info("In compute_metrics()")
    for i, input in enumerate(inputs):
        debug_tensor(f"inputs[{i}]", input)
    for i, mask in enumerate(masks):
        debug_tensor(f"masks[{i}]", mask)
    attention_mask = torch.stack([mask if isinstance(mask, torch.Tensor) else torch.from_numpy(mask) for mask in masks], dim = 0)
    input_ids = torch.stack([input if isinstance(input, torch.Tensor) else torch.from_numpy(input) for input in inputs], dim = 0)
    debug_tensor("preds", preds)
    debug_tensor("labels", labels)
    debug_tensor("attention_mask", attention_mask)
    debug_tensor("input_ids", input_ids)

    attention_mask = attention_mask.bool()
    attention_mask = attention_mask | (input_ids == IMAGE_TOKEN_INDEX)
    labels = [
        cur_labels[cur_attention_mask]
        for cur_labels, cur_attention_mask in zip(labels, attention_mask)
    ]
    labels = pad_sequence(labels, batch_first = True, padding_value = IGNORE_INDEX)

    assert preds.shape[0] == labels.shape[0], "batch size must be the same"
    batch_size = labels.shape[0]
    pred_labels = []
    gold_labels = []
    for i in range(batch_size):
        output_range = [-1, -1]
        for j in range(labels.shape[1]):
            if labels[i, j] == 78191:
                assert labels[i, j-1] == 128006 and labels[i, j+1] == 128007, "assistant token must be surrounded by start and end tokens"
                output_range[0] = j+1
        for j in range(output_range[0], labels.shape[1]):
            if labels[i, j] == 128009:
                output_range[1] = j
                break
        tcm_logger.debug(f"batch {i}: output_range={output_range}")
        cur_logits = preds[i, output_range[0]:output_range[1], :].unsqueeze(0)
        cur_outputs = cur_logits.argmax(dim = -1)
        tcm_logger.debug(f"batch {i}: cur_outputs={cur_outputs}")
        cur_labels = labels[i, output_range[0]:output_range[1]].unsqueeze(0)
        decoded_outputs = tokenizer.batch_decode(cur_outputs, skip_special_tokens=True)
        tcm_logger.debug(f"batch {i}: decoded_outputs={decoded_outputs}")
        decoded_labels = tokenizer.batch_decode(cur_labels, skip_special_tokens=True)
        tcm_logger.debug(f"batch {i}: decoded_labels={decoded_labels}")
        pred_label = extract_engagement_label(decoded_outputs[0])
        gold_label = extract_engagement_label(decoded_labels[0])
        pred_labels.append(pred_label)
        gold_labels.append(gold_label)

    # # Get predicted class by taking the argmax of logits
    # predictions = logits.argmax(axis=-1)
    # logging.info(f'predictions={predictions}')
    tcm_logger.debug(f"pred_labels={pred_labels}")
    tcm_logger.debug(f"gold_labels={gold_labels}")
    global_metrics_accumulator.update(gold_labels, pred_labels)
    # Compute accuracy
    # acc = accuracy_score(gold_labels, pred_labels)
    
    # Compute precision, recall, and F1-score
    # precision, recall, f1, _ = precision_recall_fscore_support(gold_labels, pred_labels, average='weighted')
    
    # Return metrics as a dictionary
    # return {
    #     "accuracy": acc,
    #     "precision": precision,
    #     "recall": recall,
    #     "f1": f1
    # }
    if compute_result:
        tcm_logger.debug(f"global number of eval samples={global_metrics_accumulator.length()}")
        tcm_logger.debug(f"global_metrics_accumulator.golds={global_metrics_accumulator.golds}")
        tcm_logger.debug(f"global_metrics_accumulator.preds={global_metrics_accumulator.preds}")
        result = global_metrics_accumulator.compute()
        global_metrics_accumulator.reset()
        return result
    else:
        return {}

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        # pyre-fixme[2]: Parameter must be annotated.
        data_args,
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        # pyre-fixme[4]: Attribute must be annotated.
        self.list_data_dict = list_data_dict
        self.data_path = data_path
        # pyre-fixme[4]: Attribute must be annotated.
        self.data_args = data_args
        # pyre-fixme[4]: Attribute must be annotated.
        self.length = self._get_length()

    # pyre-fixme[3]: Return type must be annotated.
    def _get_length(self):
        """Calculates the number of samples in the .jsonl file."""
        with open(self.data_path, "r") as file:
            for i, _ in enumerate(file):
                pass
        return i + 1  # pyre-fixme

    def __len__(self) -> int:
        return len(self.list_data_dict)

    # pyre-fixme[3]: Return type must be annotated.
    def _compute_lengths(self):
        """Compute and cache lengths of conversations in the dataset."""
        if hasattr(self, "length_list") and hasattr(self, "modality_length_list"):
            # Return cached values if already computed
            return self.length_list, self.modality_length_list  # pyre-fixme

        self.length_list = []
        self.modality_length_list = []
        for sample in self.list_data_dict:
            img_tokens = (
                self.data_args.image_token_len if self._has_image(sample) else 0
            )
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            self.length_list.append(cur_len + img_tokens)
            modality_len = cur_len if "image" in sample else -cur_len
            self.modality_length_list.append(modality_len)
        return self.length_list, self.modality_length_list

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def lengths(self):
        length_list, _ = self._compute_lengths()
        return length_list

    @property
    # pyre-fixme[3]: Return type must be annotated.
    def modality_lengths(self):
        _, modality_length_list = self._compute_lengths()
        return modality_length_list

    def _has_image(self, sample: dict) -> bool:  # pyre-fixme
        if "image" in sample and not str(sample["image"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        if "video" in sample and not str(sample["video"]) in [
            "",
            "None",
            "none",
            "nan",
        ]:
            return True
        return False

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        dat = sources
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        has_image = self._has_image(dat)
        if has_image:
            if "image" in dat:
                image_file = dat["image"]
                image_folder = self.data_args.image_folder
                processor_aux_list = self.data_args.image_processor_aux_list
                try:
                    image = Image.open(os.path.join(image_folder, image_file)).convert(
                        "RGB"
                    )
                except:
                    print(
                        "Not exist: ",
                        os.path.join(image_folder, image_file),
                        flush=True,
                    )
                    return self.__getitem__(0)
                image_size = image.size
            else:
                video_file = dat["video"]
                processor_aux_list = self.data_args.image_processor_aux_list
                if video_file.endswith(".gif"):
                    video_file = os.path.join(
                        self.data_args.image_folder, "gifs", video_file
                    )
                else:
                    video_file = os.path.join(self.data_args.image_folder, video_file)
                if os.path.exists(video_file):
                    try:
                        if video_file.endswith(".npy"):
                            image = np.load(video_file)
                            image_size = image[0].shape[:2]
                        elif video_file.endswith(".gif"):
                            video = Image.open(video_file)
                            image = []
                            for frame in ImageSequence.Iterator(video):
                                frame_copy = frame.copy()
                                image.append(frame_copy.convert("RGB"))
                            image_size = image[0].size
                        elif os.path.isdir(video_file):
                            files = [f for f in sorted(os.listdir(video_file))]
                            image = []
                            for file in files:
                                image.append(
                                    Image.open(os.path.join(video_file, file)).convert(
                                        "RGB"
                                    )
                                )
                            image_size = image[0].size
                        else:
                            tcm_logger.info(f"@tcm: In LazySupervisedDataset: video_file={video_file}")
                            vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
                            sample_fps = round(
                                vr.get_avg_fps() / self.data_args.video_fps
                            )
                            frame_idx = [i for i in range(0, len(vr), sample_fps)]
                            image = vr.get_batch(frame_idx).asnumpy()
                            image_size = image[0].shape[:2]
                        if self.data_args.uniform_sample:
                            num_sample = 100
                            if len(image) > num_sample:
                                interval = len(image) / float(num_sample)
                                indices = [int(interval * i) for i in range(num_sample)]
                                image = [image[idx] for idx in indices]
                    except:
                        print("fail to load video: ", video_file, flush=True)
                        return self.__getitem__(0)
                else:
                    print("Not exist: ", video_file, flush=True)
                    return self.__getitem__(0)

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    # result.paste(pil_img, (0, 0))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    # result.paste(pil_img, (0, 0))
                    return result

            if self.data_args.image_aspect_ratio != "pad":
                raise NotImplementedError("Only pad is supported for now.")

            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                if not isinstance(image_aux, Image.Image):
                    frame_list = []
                    for frame in image_aux:
                        if not isinstance(frame, Image.Image):
                            frame = Image.fromarray(frame)
                        frame_aux = expand2square(
                            frame, tuple(int(x * 255) for x in processor_aux.image_mean)
                        ).resize((target_resolution, target_resolution))
                        frame_aux = processor_aux.preprocess(
                            frame_aux, return_tensors="pt"
                        )["pixel_values"][0]
                        frame_list.append(frame_aux)
                    image_aux = torch.stack(frame_list)
                else:
                    image_aux = expand2square(
                        image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)
                    ).resize((target_resolution, target_resolution))
                    image_aux = processor_aux.preprocess(
                        image_aux, return_tensors="pt"
                    )["pixel_values"][0]
                image_aux_list.append(image_aux)

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)  # pyre-fixme
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )
        if (data_dict["labels"] != IGNORE_INDEX).sum() == 0:
            # @tcm: no assistant response given
            return self.__getitem__(0)
        # image exist in the data
        if has_image:
            data_dict["image_aux_list"] = image_aux_list  # pyre-fixme
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = 336
            processor_aux_list = self.data_args.image_processor_aux_list
            image_list = []
            for processor_aux in processor_aux_list:
                try:
                    target_resolution = processor_aux.crop_size["height"]
                except:
                    target_resolution = processor_aux.size["height"]
                image_list.append(
                    torch.zeros(
                        3,
                        target_resolution,
                        target_resolution,
                    )
                )
            data_dict["image_aux_list"] = image_list
            image_size = (crop_size, crop_size)
        data_dict["image_size"] = image_size  # pyre-fixme
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    image_token_len: int
    image_aux_token_len_list: list  # pyre-fixme
    image_position: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:  # pyre-fixme

        image_token_len = self.image_token_len
        image_aux_token_len_list = self.image_aux_token_len_list
        image_position = self.image_position

        # input_ids, labels = tuple(
        #     [instance[key] for instance in instances] for key in ("input_ids", "labels")
        # )
        input_ids = []
        for i, instance in enumerate(instances):
            # if isinstance(instance["input_ids"], torch.Tensor):
            #     debug_tensor(f"instances[{i}]['input_ids']", instance["input_ids"])
            # tcm_logger.debug(f"instances[{i}]['input_ids']: {instance['input_ids']}")
            input_ids.append(instance["input_ids"])
            # instances[0]['input_ids']: [torch.Size([154]), torch.int64, cpu]
        labels = []
        for i, instance in enumerate(instances):
            # if isinstance(instance["labels"], torch.Tensor):
            #     debug_tensor(f"instances[{i}]['labels']", instance["labels"])
            # tcm_logger.debug(f"instances[{i}]['labels']: {instance['labels']}")
            labels.append(instance["labels"])
            # instances[0]['labels']: [torch.Size([154]), torch.int64, cpu]
        max_length = self.tokenizer.model_max_length # 8192

        padding_side = self.tokenizer.padding_side

        # print_rank0("Pad token id is", self.tokenizer.pad_token_id)
        # self.tokenizer.pad_token_id: 128002
        if padding_side == "left":
            input_ids = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t,
                        (max_length - t.shape[0], 0),
                        "constant",
                        self.tokenizer.pad_token_id,
                    )
                )
                for t in input_ids
            ]
            labels = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t, (max_length - t.shape[0], 0), "constant", IGNORE_INDEX
                    )
                )
                for t in labels
            ]
        else:
            # self.tokenizer.pad_token_id: 128002
            input_ids = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t,
                        (0, max_length - t.shape[0]),
                        "constant",
                        self.tokenizer.pad_token_id,
                    )
                )
                for t in input_ids
            ]
            labels = [
                (
                    t[:max_length]
                    if t.shape[0] >= max_length
                    else torch.nn.functional.pad(
                        t, (0, max_length - t.shape[0]), "constant", IGNORE_INDEX
                    )
                )
                for t in labels
            ]

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)  # pyre-fixme
        # insert dummy image
        for i in range(len(input_ids)):
            if (input_ids[i] == IMAGE_TOKEN_INDEX).sum() == 0:
                # tcm_logger.debug(f"image_position: {image_position}")
                cur_input_ids_tmp = input_ids[i].clone()
                cur_input_ids_tmp[image_position + 1 :] = input_ids[
                    i, image_position:-1
                ]
                cur_input_ids_tmp[image_position] = IMAGE_TOKEN_INDEX
                input_ids[i] = cur_input_ids_tmp
                # tcm_logger.debug(f"cur_input_ids_tmp[:image_position+1]: {cur_input_ids_tmp[:image_position+1]}")

                cur_labels_tmp = labels[i].clone()
                cur_labels_tmp[image_position + 1 :] = labels[i, image_position:-1]
                cur_labels_tmp[image_position] = IGNORE_INDEX
                labels[i] = cur_labels_tmp
                # tcm_logger.debug(f"cur_labels_tmp[:image_position+1]: {cur_labels_tmp[:image_position+1]}")

                cur_attention_mask_tmp = attention_mask[i].clone()
                cur_attention_mask_tmp[image_position + 1 :] = attention_mask[
                    i, image_position:-1
                ]
                cur_attention_mask_tmp[image_position] = False
                attention_mask[i] = cur_attention_mask_tmp
                # tcm_logger.debug(f"cur_attention_mask_tmp[:image_position+1]: {cur_attention_mask_tmp[:image_position+1]}")
        image_sizes = [instance["image_size"] for instance in instances]
        (
            new_input_ids,
            new_labels,
            new_attention_mask,
            new_position_ids,
            im_aux_attention_masks_list,
        ) = prepare_multimodal_data(
            input_ids,
            labels,
            attention_mask,
            image_sizes,
            image_token_len,
            image_aux_token_len_list,
            max_length,
        )
        batch = dict(
            input_ids=new_input_ids,
            labels=new_labels,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            image_aux_attention_masks_list=im_aux_attention_masks_list,
        )
        batch["image_sizes"] = image_sizes
        if "image_aux_list" in instances[0]:
            image_aux_list = [instance["image_aux_list"] for instance in instances]
            image_aux_list = [
                list(batch_image_aux) for batch_image_aux in zip(*image_aux_list)
            ]
            if all(
                x is not None and x.shape == image_aux_list[0][0].shape
                for x in image_aux_list[0]
            ):
                batch["images"] = [
                    torch.stack(image_aux) for image_aux in image_aux_list
                ]
            else:
                batch["images"] = image_aux_list

        # if isinstance(batch['images'], list):
        #     for i, img_tensor in enumerate(batch['images']):
        #         if isinstance(img_tensor, torch.Tensor):
        #             debug_tensor(f"DataCollator: batch['images'][{i}]", img_tensor)
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args  # pyre-fixme
) -> Dict:  # pyre-fixme
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path_train, data_args=data_args
    )
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path_val, data_args=data_args
    )
    data_collator_kwargs = {
        "tokenizer": tokenizer,
    }

    if hasattr(data_args, "image_token_len"):
        data_collator_kwargs["image_token_len"] = data_args.image_token_len

    if hasattr(data_args, "vision_tower_aux_token_len_list"):
        data_collator_kwargs["image_aux_token_len_list"] = (
            data_args.vision_tower_aux_token_len_list
        )
    else:
        data_collator_kwargs["image_aux_token_len_list"] = [data_args.image_token_len]

    if hasattr(data_args, "image_position"):
        data_collator_kwargs["image_position"] = data_args.image_position

    data_collator = DataCollatorForSupervisedDataset(**data_collator_kwargs)  # pyre-fixme

    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator
    )


class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(
                eval_dataset = self._trainer.train_dataset,
                metric_key_prefix = "train_"
            )
            return control_copy


def train() -> None:
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    global_rank = get_global_rank()
    local_rank = get_local_rank()

    torch.distributed.barrier()
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # torch.cuda.memory._record_memory_history()

    # pyre-fixme[16]: `DataClass` has no attribute `output_model_local_path`.
    training_args.output_dir = model_args.output_model_filename
    # pyre-fixme[16]: `DataClass` has no attribute `local_dir`.
    model_args.local_dir = model_args.output_model_filename

    bnb_model_from_pretrained_args = {}

    # pyre-fixme[16]: `DataClass` has no attribute `vision_tower`.
    if model_args.vision_tower_aux_list is not None:
        # @tcm: load pre-trained longvu from checkpoint longvu_qwen2
        if "cambrian" in model_args.input_model_filename.lower() or "longvu_qwen2" in model_args.input_model_filename.lower() or "longvu_llama3_2" in model_args.input_model_filename.lower():
            if "qwen" in model_args.input_model_filename.lower():
                model = CambrianQwenForCausalLM.from_pretrained(  # pyre-fixme
                    model_args.input_model_filename,  # pyre-fixme
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),  # pyre-fixme
                    **bnb_model_from_pretrained_args,
                )
            else:
                # pyre-fixme[16]: `CambrianLlamaForCausalLM` has no attribute
                #  `from_pretrained`.
                model = CambrianLlamaForCausalLM.from_pretrained(
                    # pyre-fixme[16]: `DataClass` has no attribute `input_model_local_path`.
                    model_args.input_model_filename,
                    **bnb_model_from_pretrained_args,
                )
        else:
            raise NotImplementedError(
                f"{model_args.model_name_or_path} is not supported yet"
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.input_model_filename,
            **bnb_model_from_pretrained_args,
        )
    model.config.use_cache = False

    # pyre-fixme[16]: `DataClass` has no attribute `freeze_backbone`.
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    # pyre-fixme[16]: `DataClass` has no attribute `gradient_checkpointing`.
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.input_model_filename,
        # pyre-fixme[16]: `DataClass` has no attribute `model_max_length`.
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # pyre-fixme[16]: `DataClass` has no attribute `version`.
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    elif model_args.version == "v1":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    elif model_args.version == "phi3":
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    elif (
        model_args.version == "llama3"
        or model_args.version == "llama3_1"
        or model_args.version == "llama3_2"
    ):
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    elif model_args.version == "qwen":
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates[
            model_args.version
        ]
    else:
        if tokenizer.pad_token is None:
            print(f"Adding pad token as '<pad>'")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="<pad>"),
                tokenizer=tokenizer,
                model=model,
            )

        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]
    print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    # pyre-fixme[16]: `DataClass` has no attribute `vision_tower_aux_list`.
    if model_args.vision_tower_aux_list is not None:
        # pyre-fixme[16]: `DataClass` has no attribute `unfreeze_mm_vision_tower`.
        model_args.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        model_args.vision_tower_aux_list = json.loads(model_args.vision_tower_aux_list)
        # pyre-fixme[16]: `DataClass` has no attribute `vision_tower_aux_token_len_list`.
        model_args.vision_tower_aux_token_len_list = json.loads(
            model_args.vision_tower_aux_token_len_list
        )
        # pyre-fixme[16]: `DataClass` has no attribute `query_num_list`.
        model_args.query_num_list = json.loads(model_args.query_num_list)
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=None,  # FSDP or not, flag should be the same as None to avoid creation error
        )
        model.config.unfreeze_mm_vision_tower = training_args.unfreeze_mm_vision_tower
        vision_tower_aux_list = None
        if model_args.vision_tower_aux_list is not None:
            vision_tower_aux_list = model.get_vision_tower_aux_list()

        if not training_args.unfreeze_mm_vision_tower:
            # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(
                        dtype=torch.bfloat16, device=training_args.device  # pyre-fixme
                    )
        else:
            # vision_tower.to(device=training_args.device)
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    vision_tower_aux.to(device=training_args.device)
                # vision_tower_aux.to(dtype=torch.bfloat16, device=training_args.device)
        # data_args.image_processor = vision_tower.image_processor
        if vision_tower_aux_list is not None:
            data_args.image_processor_aux_list = [  # pyre-fixme
                vision_tower_aux.image_processor
                for vision_tower_aux in vision_tower_aux_list
            ]
        data_args.is_multimodal = True  # pyre-fixme

        model.config.image_aspect_ratio = data_args.image_aspect_ratio  # pyre-fixme
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.image_position = data_args.image_position  # pyre-fixme
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token  # pyre-fixme

        # pyre-fixme
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            # for p in model.get_model().mm_projector.parameters():
            #     p.requires_grad = True
            tune_modules = [
                "mm_projector",
                "pos_emb",
                "vision_sampler",
                "vision_sampler_layers",
                "vision_query",
                "image_newline",
            ]
            for name, param in model.named_parameters():
                if any(listed_name in name for listed_name in tune_modules):
                    param.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter  # pyre-fixme
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
        if training_args.unfreeze_mm_vision_tower:
            if vision_tower_aux_list is not None:
                for vision_tower_aux in vision_tower_aux_list:
                    for p in vision_tower_aux.parameters():
                        p.requires_grad = True

        model.config.mm_use_im_start_end = model_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.image_token_len = model_args.image_token_len = (  # pyre-fixme
            model_args.image_token_len
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr  # pyre-fixme
        model.config.mm_vision_sampler_lr = training_args.mm_vision_sampler_lr  # pyre-fixme
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr  # pyre-fixme
        training_args.use_im_start_end = model_args.mm_use_im_start_end  # pyre-fixme
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.vision_tower_aux_token_len_list = (
            data_args.vision_tower_aux_token_len_list
        ) = model_args.vision_tower_aux_token_len_list
        model.config.image_token_len = model_args.image_token_len
        model.config.is_st_sampler = model_args.is_st_sampler  # pyre-fixme
        data_args.image_token_len = model_args.image_token_len
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    tcm_logger.info(f'Total params: {total_params}')
    tcm_logger.info(f'Trainable params: {trainable_params}')

    if training_args.bf16:
        model.to(torch.bfloat16)

    # pyre-fixme
    def convert_bn_to_float(model):
        if isinstance(model, torch.nn.modules.batchnorm._BatchNorm):
            return model.float()
        for child_name, child in model.named_children():
            model.add_module(child_name, convert_bn_to_float(child))
        return model

    model = convert_bn_to_float(model)
    # with open('cambrianllama.txt', 'w', encoding='utf-8') as f:
    #     f.write(str(model))

    os.environ[f"FSDP_USE_ORIG_PARAMS"] = "true"
    # pyre-fixme[16]: `DataClass` has no attribute `fsdp_config`.
    training_args.fsdp_config["use_orig_params"] = True
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # with open('environ.json', 'w', encoding = 'utf-8') as f:
    #     json.dump(dict(os.environ), f, indent=4)

    # with open('fsdp_config.json', 'w', encoding = 'utf-8') as f:
    #     json.dump(dict(training_args.fsdp_config), f, indent=4)

    callbacks = []
    # configure TensorboardCallback to upload to manifold
    callbacks.append(
        TensorBoardCallback(
            SummaryWriter(
                log_dir=os.path.join(
                    # pyre-fixme[16]: `DataClass` has no attribute
                    #  `output_model_filename`.
                    model_args.output_model_filename,
                    TENSORBOARD_LOG_DIR_NAME,
                ),
                comment="",
                purge_step=None,
                max_queue=10,
                flush_secs=120,
                filename_suffix=str(uuid.uuid4()),
            )
        )
    )
    compute_metrics_wTokenizer = partial(compute_metrics, tokenizer=tokenizer)
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics_wTokenizer,
        # compute_metrics=compute_metrics,
        callbacks=callbacks,
        # deepspeed=training_args.deepspeed,
        **data_module,
    )
    # trainer.add_callback(CustomCallback(trainer))

    # pyre-fixme[16]: `DataClass` has no attribute `output_dir`.
#    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#        # pyre-fixme[16]: `LLaVATrainer` has no attribute `train`.
#        trainer.train(resume_from_checkpoint=True)
#    else:
#        trainer.train()

    cur_rank = int(os.environ.get('RANK', 0))
    # torch.cuda.memory._dump_snapshot(datetime.datetime.now().strftime(f'/media02/nthuy/pickle/rank_{cur_rank}_beforetrain_%d_%H%M%S.pickle'))

    trainer.train()

    # torch.cuda.memory._record_memory_history(enabled=None)
    # pyre-fixme[16]: `LLaVATrainer` has no attribute `save_state`.
    # trainer.evaluate()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        # pyre-fixme[16]: `DataClass` has no attribute `output_model_local_path`.
        output_dir=model_args.output_model_filename,
    )


if __name__ == "__main__":
    train()
