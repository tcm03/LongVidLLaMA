# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# pyre-strict

# Need to call this before importing transformers.


import copy
import datetime
import json
import os
import pathlib
import uuid
from dataclasses import dataclass, field
from logging import Logger
from typing import Dict, List, Optional, Sequence

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

import pandas as pd

from transformers.integrations import TensorBoardCallback

import logging

from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

TENSORBOARD_LOG_DIR_NAME: str = "tensorboard_logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(filename)s:%(lineno)d - %(funcName)s - %(levelname)s - %(message)s"
)

def train() -> None:

    bnb_model_from_pretrained_args = {}

    model = CambrianLlamaForCausalLM.from_pretrained(
                # pyre-fixme[16]: `DataClass` has no attribute `input_model_local_path`.
                "./checkpoints/longvu_llama3_2",
                **bnb_model_from_pretrained_args,
            )
    estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=3, num_nodes=1)



    


if __name__ == "__main__":
    train()


