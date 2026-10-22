#!/usr/bin/env bash
# Chạy toàn bộ pipeline theo thứ tự ưu tiên RQ1 -> RQ3 -> RQ4 -> thống kê.
# Mọi bước đều bỏ qua run đã có file DONE, nên nếu máy bị ngắt chỉ cần chạy lại lệnh này.
#   bash run_all.sh                       # config mặc định
#   CONFIG=configs/smoke.yaml bash run_all.sh
# Nên chạy trong tmux:  tmux new -s exp 'bash run_all.sh 2>&1 | tee -a run_all.log'
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env.sh ] && source .env.sh
# Container vast.ai thấy toàn bộ CPU của host (vd. 128) nhưng cgroup chỉ cấp ~16. Không giới hạn thì
# OpenBLAS/OpenMP/torch mở 64-128 luồng, tranh nhau và bị hãm, làm TF-IDF chậm hơn nhiều lần.
NT="${NUM_THREADS:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NT}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NT}" \
       MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NT}"
export PYTHONUNBUFFERED=1   # ghi log ngay từng dòng (khi qua `tee`, Python mặc định giữ output trong bộ đệm)
CONFIG="${CONFIG:-configs/base.yaml}"
PY="${PY:-python}"

step() { echo; echo "================ $(date '+%F %T') | $*"; }

step "Ước lượng thời gian (trước khi chạy)"
$PY -m src.utils estimate --config "$CONFIG"

step "0. Sinh tập test nhiễu (CPU, vài giây)"
$PY -m src.noise --config "$CONFIG"

step "RQ1a. TF-IDF + LR/SVM (CPU)"
$PY -m src.baseline_tfidf --config "$CONFIG" --exp rq1

step "RQ1b. Fine-tune PLM (đồng thời chấm test nhiễu và vigo_new_test)"
# Thứ tự: model base trước, CafeBERT sau cùng, để có kết quả chính sớm nhất.
for m in visobert phobert xlmr cafebert; do
  $PY -m src.train_plm --config "$CONFIG" --exp rq1 --model "$m"
  $PY -m src.utils estimate --config "$CONFIG"
done

step "RQ1c. Bảng + kiểm định RQ1 (có thể xem kết quả sớm)"
$PY -m src.stats --config "$CONFIG"

step "RQ3. Chấm bổ sung tập nhiễu còn thiếu, train augment + ma trận X/Y, báo cáo"
$PY -m src.rq3_robustness eval --config "$CONFIG"
$PY -m src.rq3_robustness train --config "$CONFIG"
$PY -m src.rq3_robustness report --config "$CONFIG"

step "RQ4. Train mốc trần trên vigo_new, chấm chéo, Cohen's kappa"
$PY -m src.rq4_cross train --config "$CONFIG"
$PY -m src.rq4_cross report --config "$CONFIG"

step "Thống kê cuối"
$PY -m src.stats --config "$CONFIG"
$PY -m src.utils estimate --config "$CONFIG"
step "XONG. Bảng ở <results_dir>/tables/"
