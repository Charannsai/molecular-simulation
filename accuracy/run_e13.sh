#!/usr/bin/env bash
# E13 drift tolerance (GPT-2 small, noise-aware fine-tune of E7). Output: results/E13_drift.json
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error TZ=Asia/Kolkata
cd "$(dirname "$0")" && $HOME/molsim-acc/.venv/bin/python -u e13_drift.py
echo "finished $(date)"
