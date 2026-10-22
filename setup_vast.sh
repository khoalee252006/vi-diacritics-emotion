#!/usr/bin/env bash
# Cài môi trường trên máy vast.ai (Ubuntu + GPU NVIDIA). Chạy từ thư mục gốc dự án:
#   bash setup_vast.sh
# Việc làm: cài Java (cho VnCoreNLP), tạo venv Python 3.12 bằng uv, cài torch CUDA 12.8 + requirements,
# tải VnCoreNLP (jar + model tách từ), kiểm tra GPU và tải trước 4 model từ Hugging Face.
set -euo pipefail
cd "$(dirname "$0")"

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

echo "== 1. Java (OpenJDK 17, headless) + curl =="
PKGS=""
command -v java >/dev/null 2>&1 || PKGS="$PKGS openjdk-17-jdk-headless"
command -v curl >/dev/null 2>&1 || PKGS="$PKGS curl"
if [ -n "$PKGS" ]; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -y
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y $PKGS
fi
java -version
JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")"
export JAVA_HOME
echo "JAVA_HOME=$JAVA_HOME"

echo "== 2. uv + venv Python 3.12 =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv .venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

echo "== 3. torch ${TORCH_VERSION} (${TORCH_INDEX}) + requirements =="
uv pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
uv pip install -r requirements.txt

echo "== 4. Ghi biến môi trường vào .env.sh (run_all.sh sẽ source) =="
cat > .env.sh <<EOF
export JAVA_HOME="$JAVA_HOME"
export HF_HOME="$HF_HOME"
export PATH="\$HOME/.local/bin:\$PATH"
export TOKENIZERS_PARALLELISM=false
source "$PWD/.venv/bin/activate"
EOF

echo "== 5. VnCoreNLP + kiểm tra tách từ =="
python - <<'EOF'
from src.utils import load_config
from src.preprocess import prepare
cfg = load_config("configs/base.yaml")
print(prepare(["ông nguyễn khắc chúc đang làm việc tại đại học quốc gia hà nội 😂"], "P0", cfg, segment=True))
EOF

echo "== 6. Kiểm tra GPU =="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "Không thấy GPU. Kiểm tra driver / chọn máy có CUDA >= 12.8."
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "GB")
x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16); print("fp16 matmul ok", (x @ x).shape)
EOF

echo "== 7. Tải trước model (theo revision trong config) =="
python - <<'EOF'
from src.utils import load_config
from transformers import AutoTokenizer, AutoModelForSequenceClassification
cfg = load_config("configs/base.yaml")
for k, m in cfg["models"].items():
    AutoTokenizer.from_pretrained(m["hf"], revision=m.get("revision"))
    AutoModelForSequenceClassification.from_pretrained(m["hf"], revision=m.get("revision"), num_labels=7)
    print("ok", k, m["hf"])
EOF
echo "Xong. Tiếp theo: bash run_all.sh"
