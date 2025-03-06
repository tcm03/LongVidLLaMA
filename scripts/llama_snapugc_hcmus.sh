
PREV_STAGE_CHECKPOINT="./checkpoints/longvu_llama3_2"
PATH_TO_JSON_TRAIN="/raid/nthuy/SnapUGC/snapugc_60s_train.json"
PATH_TO_JSON_VAL="/raid/nthuy/SnapUGC/snapugc_mini_test.json"
PATH_TO_FOLDER="/raid/nthuy/SnapUGC"
VERSION="llama3"

CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 --rdzv_endpoint=localhost:29502 --nnodes=1 \
    longvu/finetune_llama.py \
    --output_dir "/tmp/longvu/" \
    --input_model_filename $PREV_STAGE_CHECKPOINT \
    --output_model_filename "./checkpoints/cambrian_llama3_2/" \
    --model_name_or_path "longvu_llama3_2" \
    --data_path_train $PATH_TO_JSON_TRAIN \
    --data_path_val $PATH_TO_JSON_VAL \
    --image_folder $PATH_TO_FOLDER \
    --model_max_length 8192 \
    --fp16 False \
    --bf16 True \
    --log_on_each_node False \
    --logging_dir /tmp/llava/test/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_steps 625 \
    --eval_steps 500 \
    --logging_steps 5 \
    --eval_strategy "epoch" \
    --save_strategy "steps" \
    --report_to "tensorboard" \
    --save_total_limit 1 \
    --learning_rate 4.6e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --tf32 False \
    --version $VERSION \
    --mm_vision_select_layer "-2" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter False \
    --freeze_backbone True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'Dinov2Layer,SiglipEncoderLayer,LlamaDecoderLayer' \
    --gradient_checkpointing True \
    --mm_projector_type sva \
    --image_token_len 144 \
    --query_num_list "[144]" \
    --resume True \
    --lowres_token 8 \
    --video_fps 1 \
    --highres True \
    --drop_threshold 0.75 \
    --label_names labels \
    --include_inputs_for_metrics True \
    --batch_eval_metrics True \
    --torch_empty_cache_steps 1 \
    --save_only_model True
