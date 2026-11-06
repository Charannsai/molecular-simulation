"""Build the Colab package for E11 on a bigger model (addition 4): Qwen2.5-3B on a free T4, final design (test C).

    python accuracy/make_colab_bundle.py   -> docs/colab/e11_bundle.zip and docs/colab/E11_large_models.ipynb

The bundle holds only what bench_tasks.py needs (the emulator, the benchmark and the device register), so the
project itself does not have to be public. Upload the zip in the notebook's first cell, run all cells, and download
the results zip at the end; then copy its JSON files into results/E11_tasks/.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "colab"
FILES = ["accuracy/bench_tasks.py", "accuracy/chip_generate.py", "accuracy/gpt2_common.py", "params/devices.yaml"]
NL = "\n"

CELLS = [
    ("markdown", NL.join([
        "# E11 on a bigger model: question-answering on the emulated molecular chip",
        "",
        "Runs the same benchmark as on the laptop (ARC-Easy, ARC-Challenge, HellaSwag, MMLU, WikiText-2) on Qwen2.5-3B, "
        "normally and with every weight on emulated molecular devices, using the final design from the laptop tests: "
        "per-group input scaling (test C), 8-bit inputs, 12-bit outputs, digital output head.",
        "",
        "**Runtime:** Runtime > Change runtime type > **T4 GPU** (free). On an A100 you can set MODEL to Qwen2.5-7B-Instruct instead.",
        "",
        "**Steps:** Runtime > Run all. Cell 3 asks you to upload `e11_bundle.zip`. Keep the tab open (about 2 to 3 hours). "
        "The last cell downloads `e11_results.zip`; send it back and the JSON files go into `results/E11_tasks/`.",
    ])),
    ("code", NL.join(["!pip -q install aihwkit==1.2.0 transformers datasets accelerate pint pyyaml", "!nvidia-smi --query-gpu=name,memory.total --format=csv"])),
    ("code", NL.join([
        "from google.colab import files",
        "import zipfile",
        "up = files.upload()  # choose e11_bundle.zip",
        "zipfile.ZipFile(next(iter(up))).extractall('/content/work')",
        "%cd /content/work",
    ])),
    ("markdown", "Model: Qwen2.5-3B-Instruct (open, no login, fits a 16 GB T4 on the emulated chip). 200 questions per task, float16 on a T4."),
    ("code", NL.join([
        "MODEL = 'Qwen/Qwen2.5-3B-Instruct'",
        "ARGS = '--tasks arc_easy,arc_challenge,hellaswag,mmlu,wikitext --n 200 --dtype auto'",
        "import os",
        "os.environ['TRANSFORMERS_VERBOSITY'] = 'error'",
        "os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'",
    ])),
    ("code", "# 1) normal (digital) run, about 10 minutes" + NL + "!python -u accuracy/bench_tasks.py --model $MODEL --mode digital $ARGS --tag colab"),
    ("code", "# 2) final chip design (test C), chip 1, about 40 to 60 minutes" + NL
             + "!python -u accuracy/bench_tasks.py --model $MODEL --mode chip --head digital --tile-scaling --seed 0 $ARGS --tag colab_C"),
    ("code", "# 3) the same design on a second, freshly programmed chip" + NL
             + "!python -u accuracy/bench_tasks.py --model $MODEL --mode chip --head digital --tile-scaling --seed 1 $ARGS --tag colab_C"),
    ("code", "# 4) for comparison: the original design without the scaling fix" + NL
             + "!python -u accuracy/bench_tasks.py --model $MODEL --mode chip --head digital --seed 0 $ARGS --tag colab_R"),
    ("code", NL.join([
        "import shutil",
        "shutil.make_archive('/content/e11_results', 'zip', 'results/E11_tasks')",
        "files.download('/content/e11_results.zip')",
    ])),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT / "e11_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            z.write(ROOT / f, f)
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "cells": [{"cell_type": t, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in CELLS]}
    (OUT / "E11_large_models.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'e11_bundle.zip'} and {OUT / 'E11_large_models.ipynb'}")


if __name__ == "__main__":
    main()
