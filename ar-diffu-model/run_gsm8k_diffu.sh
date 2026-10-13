#!/bin/bash
# Resume: diffusion GSM8K runs only (AR steps of run_strengthen_queue.sh already done).
cd "$(dirname "$0")"
LOG=logs/strengthen.log
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
step() { echo "[$(date '+%F %T')] $*" >> "$LOG"; "$@" >> "$LOG" 2>&1 || echo "[$(date '+%F %T')] FAILED: $*" >> "$LOG"; }
echo "[$(date '+%F %T')] RESUME after reboot: diffusion GSM8K" >> "$LOG"
for m in minicpm5 dream lladamoe; do
  if ls gsm8k/$m/*.json >/dev/null 2>&1; then echo "skip $m (done)" >> "$LOG"; continue; fi
  step python3 run_ar_vs_$m.py --question-set gsm8k --temps 0.7 --ar-models qwen25-0.5b \
    --gen-tokens 256 --device mps --out-dir gsm8k/$m
done
echo "[$(date '+%F %T')] QUEUE DONE" >> "$LOG"
