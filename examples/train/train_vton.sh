#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

DATASET_ROOT="/data2/qinzijing/data/newdata"
METADATA_PATH="/data2/qinzijing/data/newdata/train_metadata.jsonl"
OUTPUT_DIR="models/wan_vton_finetuned"
DS_CONFIG="examples/train/ds_config.json"
TRAIN_SCRIPT="examples/train/train_vton.py"

accelerate launch \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --zero_stage 2 \
    --deepspeed_config_file "$DS_CONFIG" \
    --gradient_accumulation_steps 4 \
    --mixed_precision "bf16" \
    "$TRAIN_SCRIPT" \
    --stage "image" \
    --batch_size 4 \
    \
    --dataset_base_path "$DATASET_ROOT" \
    --dataset_metadata_path "$METADATA_PATH" \
    --output_path "$OUTPUT_DIR" \
    \
    --height 384 \
    --width 512 \
    --num_frames 1 \
    \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --save_steps 1000 \
    --weight_decay 0.01 \
    \
    --lora_rank 64 2>&1 | tee examples/train/train_log.txt 