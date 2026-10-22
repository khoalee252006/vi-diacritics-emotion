#!/usr/bin/env bash
# Thí nghiệm bổ sung theo nhận xét của thầy (review/17_thiet_ke_bo_sung_theo_nhan_xet.md), trên 1 máy vast >= 48 GB VRAM.
#   E1: dual + dualkl seed 45–46 (XLM-R, PhoBERT, ViSoBERT) và CafeBERT seed 42–44
#   E2: dualkl λ 0.5 và 2 (XLM-R, PhoBERT, ViSoBERT, seed 42–44)
#   chấm valid nhiễu + tập tự nhiên (src.ab_score), rồi
#   E3: Qwen2.5-72B-Instruct-AWQ qua vLLM (venv RIÊNG để không đụng torch của phần train), đúng giao thức LLM cũ.
# Dùng: cd /workspace/dap && tmux new -s rev "bash scripts/vast_rev.sh 2>&1 | tee rev_run.log"
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env.sh ] || bash setup_vast.sh
# shellcheck disable=SC1091
source .env.sh
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

NGPU=$(nvidia-smi -L | wc -l)
echo "[gpu] $NGPU card"
queue_a() {  # E1: XLM-R, ViSoBERT seed 45–46 rồi CafeBERT seed 42–44
  for m in xlmr visobert; do
    for s in 45 46; do python -m src.train_plm --exp dual --model "$m" --seed "$s"; done
  done
  for s in 42 43 44; do python -m src.train_plm --exp dual --model cafebert --seed "$s"; done
}
queue_b() {  # MỌI run PhoBERT ở cùng một hàng đợi (tránh 2 tiến trình cùng ghi cache VnCoreNLP), rồi E2
  for s in 45 46; do python -m src.train_plm --exp dual --model phobert --seed "$s"; done
  for m in xlmr phobert visobert; do python -m src.train_plm --exp dual_lam --model "$m"; done
}
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  if [ "$NGPU" -ge 2 ]; then   # 2 card: chạy song song, mỗi hàng đợi một card
    CUDA_VISIBLE_DEVICES=0 queue_a > rev_queue_a.log 2>&1 &
    PA=$!
    CUDA_VISIBLE_DEVICES=1 queue_b > rev_queue_b.log 2>&1 &
    PB=$!
    wait $PA; wait $PB
  else
    queue_a; queue_b
  fi
  CUDA_VISIBLE_DEVICES=0 python -m src.ab_score
  echo "XONG TRAIN"
fi

# ---- E3: LLM mở lớn, venv riêng (SKIP_LLM=1 khi máy < 48 GB: chạy E3 trên máy khác bằng SKIP_TRAIN=1)
if [ "${SKIP_LLM:-0}" = "1" ]; then
  tar --exclude='*/checkpoint' -czf rev_results.tgz results/dual results/dual_lam && md5sum rev_results.tgz
  echo "XONG TẤT CẢ (bỏ qua LLM)"; exit 0
fi
deactivate 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"   # uv do setup_vast.sh cài
if [ ! -x /workspace/vllm_env/bin/vllm ]; then
  uv venv /workspace/vllm_env --python 3.12
  uv pip install --python /workspace/vllm_env/bin/python vllm pandas numpy pyyaml requests openpyxl scikit-learn
fi
# shellcheck disable=SC1091
source /workspace/vllm_env/bin/activate
PORT=8000
serve() {  # $1 = model, $2 = log; như vast_llm.sh: thử --generation-config vllm, bản không nhận cờ thì chạy lại không cờ
  local extra=(--generation-config vllm)
  for attempt in 1 2; do
    vllm serve "$1" --max-model-len 4096 --gpu-memory-utilization 0.95 --seed 0 --port $PORT --tensor-parallel-size "$NGPU" \
         "${extra[@]}" > "$2" 2>&1 &
    SERVER=$!
    for i in $(seq 1 360); do   # tối đa 60 phút (tải ~40 GB)
      if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then echo "[vllm] $1 sẵn sàng"; return 0; fi
      if ! kill -0 $SERVER 2>/dev/null; then break; fi
      sleep 10
    done
    kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true
    if [ $attempt -eq 1 ] && grep -q "generation-config\|generation_config" "$2"; then
      echo "[vllm] không nhận --generation-config, chạy lại không cờ"; extra=()
    else
      echo "[vllm] $1 không khởi động được, xem $2"; tail -20 "$2"; return 1
    fi
  done
  return 1
}
for M in "Qwen/Qwen2.5-72B-Instruct-AWQ" "Qwen/Qwen2.5-32B-Instruct-AWQ"; do   # 32B chỉ là dự phòng (chốt trước)
  TAG=$(echo "$M" | sed 's/[^A-Za-z0-9._-]/_/g')
  mkdir -p results/llm
  if serve "$M" "results/llm/vllm_${TAG}.log"; then
    python -m src.llm_eval run --base-url "http://localhost:$PORT/v1" --model "$M" --key-env "" --workers 8 --limit 5 --tag "pilot_${TAG}"
    python -m src.llm_eval run --base-url "http://localhost:$PORT/v1" --model "$M" --key-env "" --workers 32 --tag "$TAG"
    kill $SERVER; wait $SERVER 2>/dev/null || true
    echo "[xong] $M"
    break
  fi
done
mkdir -p results/llm_pilot && mv results/llm/pilot_* results/llm_pilot/ 2>/dev/null || true

# ---- gói kết quả (không checkpoint) để tải về
tar --exclude='*/checkpoint' -czf rev_results.tgz results/dual results/dual_lam results/llm results/llm_pilot rev_run.log 2>/dev/null || \
tar --exclude='*/checkpoint' -czf rev_results.tgz results/dual results/dual_lam results/llm results/llm_pilot
md5sum rev_results.tgz
echo "XONG TẤT CẢ"
