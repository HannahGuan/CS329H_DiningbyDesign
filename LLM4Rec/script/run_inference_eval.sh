#!/bin/bash
# Run inference on the last 20 users evaluation set
# Usage: bash run_inference_eval.sh [adapter_path] [result_output_path]
# Example: bash run_inference_eval.sh ./outputs/small_test/final_model results/inference_results.json

ADAPTER_PATH=${1:-"./outputs/small_test/final_model"}
RESULT_PATH=${2:-"results/inference_eval_last20.json"}

echo "=========================================="
echo "Running Inference on Evaluation Set"
echo "=========================================="
echo "Adapter path: ${ADAPTER_PATH}"
echo "Evaluation data: data/dpo_eval_last20.jsonl (20 users)"
echo "Result output: ${RESULT_PATH}"
echo "=========================================="
echo ""

# Create results directory if it doesn't exist
mkdir -p $(dirname ${RESULT_PATH})

# Set GPU
export CUDA_VISIBLE_DEVICES=4

# Run inference on all 20 evaluation samples
# python inference_dpo.py \
#     --base_model Qwen/Qwen2.5-3B-Instruct \
#     --adapter_path ${ADAPTER_PATH} \
#     --dpo_data data/dpo_eval_last20.jsonl \
#     --num_samples 20 \
#     --use_last \
#     --output_file ${RESULT_PATH}

echo ""
echo "=========================================="
echo "Inference complete!"
echo "=========================================="

# Analyze results
echo ""
echo "=========================================="
echo "Analyzing Results"
echo "=========================================="
python analyze_inference_results.py --results_file ${RESULT_PATH}

echo ""
echo "=========================================="
echo "All Done!"
echo "Results saved to: ${RESULT_PATH}"
echo "=========================================="

