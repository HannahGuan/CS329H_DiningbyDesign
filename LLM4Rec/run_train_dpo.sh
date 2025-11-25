#!/bin/bash

# Shell script to run train_dpo.py with proper setup
# Usage: bash run_train_dpo.sh [gpu_id]
# Example: bash run_train_dpo.sh 0

# GPU selection (default to GPU 0 if not specified)
GPU_ID=${1:-0}

echo "=========================================="
echo "DPO Training Script"
echo "=========================================="
echo "GPU ID: ${GPU_ID}"
echo "Model: Qwen/Qwen2.5-3B-Instruct"
echo "Data: data/dpo_preference_dataset_small.jsonl"
echo "Output: ./dpo_model_output"
echo "=========================================="
echo ""

# Set CUDA device
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Set Wandb API key
export WANDB_API_KEY=b53a8b344440d37f80e675cd93858227bab887a7

# Optional: Set HuggingFace token if needed (uncomment and add your token)
# export HUGGINGFACE_TOKEN=your_token_here

# Run training
echo "Starting training..."
python train_dpo.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Training completed successfully!"
    echo "=========================================="
    echo "Model saved to: ./dpo_model_output/final_model"
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed with error code $?"
    echo "=========================================="
    exit 1
fi

