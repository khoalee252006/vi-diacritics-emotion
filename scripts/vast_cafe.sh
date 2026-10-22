#!/usr/bin/env bash
# Nhận xét thầy (2026-10-21): CafeBERT đủ 5 seed cho phần phương pháp — seed 45, 46 cho model thường (rq1 P0),
# không dấu (strip PS), nhân đôi (dual) và dual-view (dualkl). Cùng công thức, cùng bộ phần mềm (setup_vast.sh).
# 2 card: seed 45 trên card 0, seed 46 trên card 1. Sau đó ab_score chấm valid nhiễu + tập tự nhiên cho run dual*.
# Dùng: cd /workspace/dap && tmux new -s cafe "bash scripts/vast_cafe.sh 2>&1 | tee cafe_run.log"
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env.sh ] || bash setup_vast.sh
# shellcheck disable=SC1091
source .env.sh
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
NGPU=$(nvidia-smi -L | wc -l)
echo "[gpu] $NGPU card"
one_seed() {  # $1 = seed
  python -m src.train_plm --exp rq1 --model cafebert --prep P0 --seed "$1"
  python -m src.train_plm --exp strip --model cafebert --seed "$1"
  python -m src.train_plm --exp dual --model cafebert --seed "$1"
}
if [ "$NGPU" -ge 2 ]; then
  CUDA_VISIBLE_DEVICES=0 one_seed 45 > cafe_s45.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=1 one_seed 46 > cafe_s46.log 2>&1 &
  P2=$!
  wait $P1; wait $P2
else
  one_seed 45 > cafe_s45.log 2>&1; one_seed 46 > cafe_s46.log 2>&1
fi
CUDA_VISIBLE_DEVICES=0 python -m src.ab_score
tar --exclude='*/checkpoint' -czf cafe_results.tgz results/rq1 results/strip results/dual cafe_run.log cafe_s45.log cafe_s46.log 2>/dev/null || \
tar --exclude='*/checkpoint' -czf cafe_results.tgz results/rq1 results/strip results/dual
md5sum cafe_results.tgz
echo "XONG TẤT CẢ"
