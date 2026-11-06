#!/usr/bin/env bash
# Accuracy environment inside WSL2 Ubuntu (AIHWKit has no Windows build).
#   wsl -d Ubuntu -- bash /mnt/c/.../accuracy/setup_wsl.sh
# The environment lives in the Linux home (~/molsim-acc): much faster than /mnt/c.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p ~/molsim-acc && cd ~/molsim-acc
[ -d .venv ] || uv venv -p 3.12 -q
uv pip install -q --index-url https://download.pytorch.org/whl/cu128 torch torchvision
uv pip install -q aihwkit==1.2.0 transformers datasets
.venv/bin/python -c 'import torch, aihwkit; print("torch", torch.__version__, "| aihwkit", aihwkit.__version__, "| cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
