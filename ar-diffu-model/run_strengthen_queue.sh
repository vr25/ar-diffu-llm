#!/bin/bash
# GPU queue for paper-strengthening experiments (runs sequentially).
cd "$(dirname "$0")"
LOG=logs/strengthen.log
mkdir -p logs
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
step() { echo "[$(date '+%F %T')] $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || echo "[$(date '+%F %T')] FAILED: $*" >> "$LOG"; }

# 1. AR-only (fast)
step python3 run_ar_only.py --ar-models qwen25-0.5b qwen25-1.5b qwen25-3b --question-set gsm8k --temps 0.7 --gen-tokens 256
step python3 run_ar_only.py --ar-models qwen25-0.5b qwen25-1.5b qwen25-3b --question-set gsm8k --temps 0.7 --gen-tokens 512
step python3 run_ar_only.py --ar-models smollm2-360m smollm2-1.7b --question-set reasoning --temps 0.5 0.7 0.9 --gen-tokens 40
step python3 run_ar_only.py --ar-models smollm2-360m smollm2-1.7b --question-set reasoning --temps 0.5 0.7 0.9 --gen-tokens 150
step python3 run_ar_only.py --ar-models qwen25-0.5b qwen25-1.5b qwen25-3b --question-set reasoning --temps 0 --gen-tokens 150

# 2. Diffusion on GSM8K (slow)
for m in minicpm5 dream lladamoe; do
  step python3 run_ar_vs_$m.py --question-set gsm8k --temps 0.7 --ar-models qwen25-0.5b \
    --gen-tokens 256 --device mps --out-dir gsm8k/$m
done
echo "[$(date '+%F %T')] QUEUE DONE" >> "$LOG"
