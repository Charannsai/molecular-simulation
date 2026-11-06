#!/usr/bin/env bash
# E11: task accuracy of Qwen2.5 models on emulated molecular crossbar tiles.
# Run inside WSL2 or Linux from the repository root, after accuracy/setup_wsl.sh:
#   bash accuracy/run_e11.sh [diag|exam|size|bits|rows|all]
# Every run skips itself when its result file already exists, so the script can be restarted.
set -uo pipefail
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PY:-$HOME/molsim-acc/.venv/bin/python -u}"
OUT=results/E11_tasks
M=Qwen/Qwen2.5-0.5B-Instruct
what="${1:-all}"
run() {  # tag, then arguments for bench_tasks.py
  local tag=$1; shift
  ls "$OUT"/*_"$tag".json >/dev/null 2>&1 && { echo "skip $tag"; return; }
  echo "=== $tag  $(date)"
  $PY accuracy/bench_tasks.py --tag "$tag" "$@" || echo "FAILED $tag"
}
S="--model $M --tasks arc_easy,mmlu,wikitext --n 200"
if [[ $what == diag || $what == all ]]; then   # diagnostics: what causes the loss
  run d200 $S --mode digital
  run R_full $S --mode chip --head digital --seed 0
  run A_circuits_only $S --mode chip --head digital --seed 0 --ideal-devices
  run B_molecules_only $S --mode chip --head digital --seed 0 --perfect-converters
  run C_tile_scaling $S --mode chip --head digital --seed 0 --tile-scaling
fi
X="--model $M --tasks arc_easy,arc_challenge,hellaswag,mmlu,wikitext --n 2000"
if [[ $what == exam || $what == all ]]; then   # larger sets: up to 2,000 questions per task
  run X2000_digital $X --mode digital
  run X2000_C $X --mode chip --head digital --seed 0 --tile-scaling
fi
if [[ $what == rows || $what == all ]]; then   # 512-row tiles
  run X2000_C_R512 $X --mode chip --head digital --seed 0 --tile-scaling --rows 512
fi
if [[ $what == size || $what == all ]]; then   # model size (3B needs about 16 GB of GPU memory)
  for s in 1.5B 3B; do
    Q="--model Qwen/Qwen2.5-$s-Instruct --tasks arc_easy,mmlu,wikitext --n 200 --dtype auto"
    run "size_${s}_digital" $Q --mode digital
    run "size_${s}_first" $Q --mode chip --head digital --seed 0
    run "size_${s}_C" $Q --mode chip --head digital --seed 0 --tile-scaling
  done
fi
if [[ $what == bits || $what == all ]]; then   # converter resolution
  B="$S --mode chip --head digital --seed 0 --tile-scaling"
  for ib in 6 7 10; do run "BITS_in${ib}_out12" $B --input-bits $ib; done
  for ob in 8 9 10 14; do run "BITS_in8_out${ob}" $B --output-bits $ob; done
fi
echo "finished $(date)"
