#!/bin/bash

# Paths to model and data
PREV_STAGE_CHECKPOINT="./checkpoints/longvu_qwen2"
OUTPUT_MODEL_DIR="./results"
DATA_PATH="/content/drive/MyDrive/Thesis/EnTube"

# Hyperparameters
NUM_EPOCHS=10
BATCH_SIZE=1  # Reduced batch size
LEARNING_RATE=1e-4  # Increased learning rate
MAX_LENGTH=8192

PYTHONFAULTHANDLER=1 CUDA_VISIBLE_DEVICES=0 python \
longvu/entube_finetune.py \
--input_model_filename $PREV_STAGE_CHECKPOINT \
--output_model_filename $OUTPUT_MODEL_DIR \
--data_path $DATA_PATH \
--num_train_epochs $NUM_EPOCHS \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--learning_rate $LEARNING_RATE \
--model_max_length $MAX_LENGTH \
--freeze_backbone True
