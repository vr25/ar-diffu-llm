#!/bin/bash
# Diffusion token-budget sweep (reasoning, T=0.7, k in {20,60,100}).
# Waits until the open-weights-survey GPU jobs finish, then runs.
cd "$(dirname "$0")"
LOG=logs/diffu_sweep.log
mkdir -p logs sweep
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

busy() { pgrep -f "exp[0-9]_[a-z_]*\.py|run_when_idle|run_exp" >/dev/null; }

echo "[$(date '+%F %T')] waiting for open-weights-survey jobs..." >> "$LOG"
idle=0
while [ $idle -lt 2 ]; do
  if busy; then idle=0; else idle=$((idle+1)); fi
  sleep 300
done
echo "[$(date '+%F %T')] GPU free, starting sweep" >> "$LOG"

for k in 20 60 100; do
  for m in lladamoe dream minicpm5; do
    # MiniCPM5 only takes max_length (prompt+answer = k+8); at k=20 that is shorter than the prompt.
    [ "$m" = minicpm5 ] && [ $k -lt 40 ] && continue
    echo "[$(date '+%F %T')] $m k=$k" >> "$LOG"
    python3 run_ar_vs_$m.py --question-set reasoning --temps 0.7 --ar-models qwen25-0.5b \
      --gen-tokens $k --device mps --out-dir sweep/${m}_k$k >> "$LOG" 2>&1 \
      || echo "[$(date '+%F %T')] FAILED $m k=$k" >> "$LOG"
  done
done
echo "[$(date '+%F %T')] SWEEP DONE" >> "$LOG"
