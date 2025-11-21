#!/bin/bash
#SBATCH --partition=defq
#SBATCH --job-name=interactive
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-gpu=8
#SBATCH --output=/home/justin/CS329H_DiningbyDesign/%x.out
#SBATCH --time=02:00:00
#SBATCH --account=liquidai


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
    --dtype bfloat16 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7
    &

echo "vLLM PID: $SERVER_PID"


# Poll readiness
for i in {1..120}; do
if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "✅ vLLM is up."
    break
fi
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "❌ vLLM crashed. Last 100 lines:"
    tail -n 100 vllm_server.log
    exit 1
fi
sleep 5
done

# Smoke test
echo "Sending a test prompt..."
curl http://localhost:${PORT}/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello, are you running correctly?"}],
    "temperature": 0.3,
    "min_p": 0.15,
    "repetition_penalty": 1.05,
    "max_tokens": 100
}' || echo "❌ Prompt failed"


# generate data
python generate_profiles.py --profile_type user

