#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=interactive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --output=/home/justin/CS329H_DiningbyDesign/%x_restaurant.out
#SBATCH --time=12:00:00



export PORT=$(( 8600 + (${SLURM_JOBID:-0} % 300) ))

echo "Host: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "JobID: ${SLURM_JOBID:-unset}"
echo "Port: $PORT"

if ss -ltn | awk '{print $4}' | grep -q ":$PORT$"; then
  echo "❌ Port $PORT is already in use"; exit 1
fi

export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"


# set up the environment - using company's cluster env
cd /home/justin/idle
source .venv/bin/activate

export TIKTOKEN_CACHE_DIR=$HOME/.cache/tiktoken-rs
echo "VLLM SERVER STARTING..."
srun vllm serve /lambdafs/pretrained/gpt-oss-120b \
    --served-model-name gpt-oss-120b \
    --host 0.0.0.0 \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7 \
    --dtype bfloat16 \
    &


echo "Waiting for vLLM server to be ready..."
start_time=$(date +%s)
timeout_duration=900  # 15 minutes in seconds

while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout_duration ]; then
        echo "Timeout: vLLM server did not become ready within 15 minutes"
        echo "Killing vLLM server and exiting..."
        pkill -f "vllm serve"
        exit 1
    fi
    
    sleep 10
    echo "Waiting for vLLM server... (${elapsed}/${timeout_duration}s elapsed)"
done
echo "vLLM server is ready!"

# generate data
python /home/justin/CS329H_DiningbyDesign/profile_generation/generate_profiles.py --profile_type restaurant
