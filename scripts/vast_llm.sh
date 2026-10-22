#!/usr/bin/env bash
# Phần 2, model mở trên vast.ai (RTX 4090 24 GB): dựng vLLM (OpenAI-compatible) rồi chạy src.llm_eval
# với ĐÚNG giao thức đã dùng cho Gemini: cùng prompt, temperature 0, max_tokens 8, zero-shot và 7-shot,
# test sạch / bỏ dấu 50% / bỏ dấu 100%. Kết quả: results/llm/<tên model>/responses.jsonl (+ log vllm_*.log).
#
# Dùng:  cd /workspace/dap && bash scripts/vast_llm.sh
# Model mặc định (không cần xin quyền):
#   Qwen2.5-7B-Instruct (model mở đa ngôn ngữ chính);
#   Qwen2.5-3B-Instruct + Arcee-VyLinh (3B, tinh chỉnh tiếng Việt TỪ Qwen2.5-3B) = cặp đối chứng sạch cho câu hỏi
#   "tinh chỉnh tiếng Việt có làm bền hơn khi mất dấu không" (cùng kiến trúc, cùng cỡ).
# Vistral-7B-Chat (repo gated) chỉ chạy khi:  export INCLUDE_VISTRAL=1 HF_TOKEN=hf_xxx
set -euo pipefail
cd "$(dirname "$0")/.."
# Image vastai/pytorch: Python + torch ở /venv/main (lệnh `python` trần không có trên PATH)
if [ -f /venv/main/bin/activate ]; then set +u; source /venv/main/bin/activate; set -u; fi

MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-3B-Instruct" "arcee-ai/Arcee-VyLinh")
if [ "${INCLUDE_VISTRAL:-0}" = "1" ]; then MODELS+=("Viet-Mistral/Vistral-7B-Chat"); fi
PORT=8000
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

if [ "${INCLUDE_VISTRAL:-0}" = "1" ] && [ -z "${HF_TOKEN:-}" ]; then echo "THIẾU HF_TOKEN (cần cho Vistral)"; exit 1; fi
python -c "import vllm" 2>/dev/null || pip install -q vllm
pip install -q pandas numpy pyyaml requests openpyxl scikit-learn

serve() {  # $1 = model, $2 = file log; thử --generation-config vllm, nếu bản vLLM không hỗ trợ thì chạy lại không có cờ
  local m="$1" log="$2" extra=(--generation-config vllm)
  for attempt in 1 2; do
    vllm serve "$m" --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 0 \
         --port $PORT "${extra[@]}" > "$log" 2>&1 &
    SERVER=$!
    for i in $(seq 1 180); do          # chờ tối đa 30 phút (tải model ~15 GB)
      if curl -sf "http://localhost:$PORT/v1/models" > /dev/null; then echo "[vllm] $m sẵn sàng"; return 0; fi
      if ! kill -0 $SERVER 2>/dev/null; then break; fi
      sleep 10
    done
    kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true
    if [ $attempt -eq 1 ] && grep -q "generation-config\|generation_config" "$log"; then
      echo "[vllm] bản này không nhận --generation-config, chạy lại không có cờ"; extra=()
    else
      echo "[vllm] KHÔNG khởi động được $m, xem $log"; tail -20 "$log"; exit 1
    fi
  done
}

for M in "${MODELS[@]}"; do
  TAG=$(echo "$M" | sed 's/[^A-Za-z0-9._-]/_/g')
  serve "$M" "results/llm/vllm_${TAG}.log"
  # Chạy thử 5 câu (tag riêng) để bắt lỗi chat template / định dạng trước khi chạy hết
  python -m src.llm_eval run --base-url "http://localhost:$PORT/v1" --model "$M" --key-env "" \
         --workers 8 --limit 5 --tag "pilot_${TAG}"
  python -m src.llm_eval run --base-url "http://localhost:$PORT/v1" --model "$M" --key-env "" \
         --workers 32 --tag "$TAG"
  kill $SERVER; wait $SERVER 2>/dev/null || true
  echo "[xong] $M"
done
mkdir -p results/llm_pilot && mv results/llm/pilot_* results/llm_pilot/ 2>/dev/null || true
echo "XONG TẤT CẢ"
