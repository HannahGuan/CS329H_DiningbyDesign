#!/bin/bash
# Example: Train on small dataset using 2 GPUs
# Usage: bash train_small_multigpu.sh
# Note: Requires accelerate to be installed (pip install accelerate)

# Specify which GPUs to use
export CUDA_VISIBLE_DEVICES=0,1

echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Launching multi-GPU training with accelerate..."
echo ""

# Launch with accelerate for proper distributed training
accelerate launch --num_processes=2 train_dpo.py \
    --data_path data/dpo_preference_dataset_small.jsonl \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./outputs/small_test_multigpu \
    --learning_rate 5e-5 \
    --batch_size 2 \
    --num_epochs 3 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --test_size 0.2 \
    --use_4bit \
    --logging_steps 2 \
    --eval_steps 10 \
    --save_steps 50 \
    --use_wandb \
    --wandb_project LLM4Rec-DPO \
    --wandb_run_name small-test-2gpus

echo ""
echo "Training complete!"

