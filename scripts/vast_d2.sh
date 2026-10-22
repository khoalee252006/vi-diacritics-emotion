#!/usr/bin/env bash
# D2 (review/09_thiet_ke_dang_la.md) trên vast.ai RTX 4090: cùng bộ phần mềm với RQ1 (setup_vast.sh: torch 2.9.1 cu128).
# Dùng: cd /workspace/dap && bash scripts/vast_d2.sh   (chạy trong tmux; log: d2_run.log)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env.sh ] || bash setup_vast.sh
# shellcheck disable=SC1091
source .env.sh
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for m in xlmr phobert visobert; do
  python -m src.train_plm --exp tgt --model "$m"
done
python -m src.tgt_eval score
python -m src.tgt_eval analyze
echo "XONG D2"
