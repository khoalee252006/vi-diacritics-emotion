#!/usr/bin/env bash
# Phương pháp A + B (review/12_thiet_ke_phuong_phap_AB.md) trên vast RTX 4090, cùng bộ phần mềm với RQ1 (setup_vast.sh).
# Dùng: cd /workspace/dap && bash scripts/vast_ab.sh   (chạy trong tmux; log: ab_run.log)
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env.sh ] || bash setup_vast.sh
# shellcheck disable=SC1091
source .env.sh
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for m in xlmr phobert visobert; do
  python -m src.train_plm --exp strip_ck --model "$m"
done
for m in xlmr phobert visobert; do
  python -m src.train_plm --exp dual --model "$m"
done
python -m src.ab_score
echo "XONG AB"
