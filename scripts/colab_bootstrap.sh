#!/usr/bin/env bash
# Bootstrap a Google Colab (or any fresh Ubuntu + NVIDIA) session for MexQL.
#
#   bash scripts/colab_bootstrap.sh            # inside an existing checkout
#   REPO_URL=https://github.com/<you>/MexQL.git bash <(curl -sSL <raw url of this file>)
#
# Installs uv, creates a Python 3.11 environment with the GPU extras (vLLM picks the
# PyTorch build matching the installed CUDA driver), and downloads the BIRD dev set.
# Set WITH_TRAIN=1 to also download the train set (~9 GB download, ~32 GB on disk).
set -euo pipefail

REPO_URL="${REPO_URL:-}"
WITH_TRAIN="${WITH_TRAIN:-0}"
WORKDIR="${WORKDIR:-/content}"

if [ ! -f pyproject.toml ]; then
  if [ -z "$REPO_URL" ]; then
    echo "run from the repository root or set REPO_URL" >&2
    exit 1
  fi
  cd "$WORKDIR"
  [ -d MexQL ] || git clone "$REPO_URL" MexQL
  cd MexQL
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python 3.11 .venv --allow-existing
# vLLM's wheel pins its own torch; --torch-backend=auto selects the CUDA build for this driver.
uv pip install --python .venv/bin/python -e ".[hf,gpu,dev]" --torch-backend=auto

.venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| vllm', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python scripts/download_data.py --dev --train-meta-only
if [ "$WITH_TRAIN" = "1" ]; then
  .venv/bin/python scripts/download_data.py --train
fi
echo "bootstrap done; run scripts with .venv/bin/python scripts/<name>.py ..."
