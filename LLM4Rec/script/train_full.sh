#!/bin/bash
# Example: Train on full dataset (17,517 users) with standard settings
# Usage: bash train_full.sh [gpu_id]
# Example: bash train_full.sh 0  (uses GPU 0)
#          bash train_full.sh 1  (uses GPU 1)

# Set GPU (default to GPU 0 if not specified)
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "Using GPU: ${GPU_ID}"
echo ""

python train_dpo.py \
    --data_path data/dpo_preference_dataset.jsonl \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./outputs/full_training \
    --learning_rate 5e-5 \
    --batch_size 4 \
    --num_epochs 3 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --test_size 0.2 \
    --use_4bit \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --save_total_limit 3 \
    --use_wandb \
    --wandb_project LLM4Rec-DPO \
    --wandb_run_name full-training-17k

