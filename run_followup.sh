#!/usr/bin/env bash
# Đợt bổ sung sau pipeline chính 20 epoch (chạy trên cùng máy vast). Thứ tự:
#   1. B1 (tái lập cấu hình bài gốc) + B2 (dò lr)                       -> results/b1_*, results/b2_*
#   2. Model không dấu (PS), 20 epoch                                    -> results/strip
#   3. 30 epoch: RQ1 trước, rồi model không dấu 30 epoch, rồi phần còn lại -> results_e30/
# Bỏ qua run đã có DONE, nên nếu máy bị ngắt chỉ cần chạy lại lệnh này.
#   bash run_followup.sh 2>&1 | tee -a run_followup.log
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env.sh ] && source .env.sh
NT="${NUM_THREADS:-8}"   # xem giải thích trong run_all.sh
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NT}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NT}" \
       MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NT}" PYTHONUNBUFFERED=1
PY="${PY:-python}"
C20=configs/base.yaml
C30=configs/e30.yaml

step() { echo; echo "================ $(date '+%F %T') | $*"; }

if ! grep -q "XONG" run_all.log 2>/dev/null; then
  echo ">>> Pipeline chính (run_all.log) chưa có dòng XONG. Chờ chạy xong rồi mới chạy script này."; exit 1
fi

for e in b1_visobert_paper b1_phobert_smtce b2_lr1e5 b2_lr3e5; do
  step "1. Bổ sung: $e"
  $PY -m src.train_plm --config configs/extra.yaml --exp "$e"
done

step "2. Model không dấu (PS), 20 epoch"
$PY -m src.baseline_tfidf --config $C20 --exp strip
$PY -m src.train_plm --config $C20 --exp strip
$PY -m src.restore_eval --config $C20 --restored results/noise/restored || echo ">>> restore_eval (20 epoch) lỗi"
$PY -m src.gate_fusion --config $C20 || echo ">>> gate_fusion (20 epoch) lỗi, bỏ qua, chạy tiếp"

step "3a. 30 epoch: sinh nhiễu + RQ1"
$PY -m src.noise --config $C30
$PY -m src.baseline_tfidf --config $C30 --exp rq1
for m in visobert phobert xlmr cafebert; do
  $PY -m src.train_plm --config $C30 --exp rq1 --model "$m"
done

step "3b. 30 epoch: model không dấu (PS)"
$PY -m src.baseline_tfidf --config $C30 --exp strip
$PY -m src.train_plm --config $C30 --exp strip

step "3c. 30 epoch: phần còn lại (RQ3, RQ4, thống kê; RQ1 đã xong nên được bỏ qua)"
CONFIG=$C30 bash run_all.sh

step "4. Kết hợp theo tỉ lệ dấu (30 epoch) + chọn 20/30 epoch theo valid"
$PY -m src.restore_eval --config $C30 --restored results/noise/restored || echo ">>> restore_eval (30 epoch) lỗi"
$PY -m src.gate_fusion --config $C30 || echo ">>> gate_fusion (30 epoch) lỗi"
$PY -m src.select_epochs --a $C20 --b $C30 || echo ">>> select_epochs lỗi"
step "XONG ĐỢT BỔ SUNG"
