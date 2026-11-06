#!/usr/bin/env bash
# E7 full pipeline inside WSL: fine-tune (digital, then noise-aware), then evaluate every condition.
#   bash accuracy/run_e7.sh [train|eval|all]
set -euo pipefail
cd "$(dirname "$0")"
PY="$HOME/molsim-acc/.venv/bin/python -u"
STEPS=1500
what="${1:-all}"
if [[ "$what" == train || "$what" == all ]]; then
  $PY e7_train.py --mode digital --steps $STEPS
  $PY e7_train.py --mode hwa --steps $STEPS
fi
if [[ "$what" == eval || "$what" == all ]]; then
  $PY e7_eval.py
fi
