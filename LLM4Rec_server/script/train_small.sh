#!/bin/bash
# Example: Train on small dataset (100 users) for quick testing
# Usage: bash train_small.sh [gpu_id]
# Example: bash train_small.sh 0  (uses GPU 0)
#          bash train_small.sh 1  (uses GPU 1)

# Set GPU (default to GPU 0 if not specified)
export CUDA_VISIBLE_DEVICES=4

# Set Wandb API key
export WANDB_API_KEY=b53a8b344440d37f80e675cd93858227bab887a7

python train_dpo.py \
    --data_path data/dpo_preference_dataset_small.jsonl \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --output_dir ./outputs/small_test \
    --learning_rate 5e-5 \
    --batch_size 4 \
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
    --wandb_run_name small-test-100users

