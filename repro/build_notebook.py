"""Writes notebooks/multiPI_TransBTS_colab.ipynb (kept as code so the notebook is reviewable)."""
import json
import sys

REPO_URL = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/mrm2025011LK/multiPI-TransBTS.git"
BRANCH = sys.argv[2] if len(sys.argv) > 2 else "reproduction"
OUT = sys.argv[3] if len(sys.argv) > 3 else "notebooks/multiPI_TransBTS_colab.ipynb"
PATCH = open(sys.argv[4]).read() if len(sys.argv) > 4 else None  # standalone flavour: upstream + patch

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(True)})


md("""
# multiPI-TransBTS: reproduction on Google Colab

Paper: *multiPI-TransBTS: A Multi-Path Learning Framework for Brain Tumor Image Segmentation Based on Multi-Physical Information* (arXiv:2409.12167).
Official code: https://github.com/JoetheReindeer/multiPI_TransBTS

This notebook
1. checks the released code without any data (split lists, Hausdorff metric, model size),
2. downloads BraTS2020 and preprocesses it exactly as the released code does,
3. trains the released model with the released settings (baseline),
4. trains the proposed variant (one changed parameter: the false-negative weight of the ET loss),
5. reports both the released metrics and standard metrics, on the official test slices and on patients never seen in training.

**Runtime → Change runtime type → T4 GPU** before running. Run the cells top to bottom.
""")

code("""
# ---------------- configuration ----------------
PRESET = "scaled"   # "scaled": fits one free Colab T4 session.  "full": the paper's setting (all 369 patients, 100 epochs).

if PRESET == "scaled":
    N_SUBSET  = 80    # patients the official split lists are restricted to
    N_HELDOUT = 30    # further patients kept completely out of training (unseen-patient test)
    EPOCHS    = 15    # upper limit; reduced automatically in section 3 to fit TRAIN_MINUTES_PER_RUN
    TRAIN_MINUTES_PER_RUN = 70
else:
    N_SUBSET, N_HELDOUT, EPOCHS, TRAIN_MINUTES_PER_RUN = None, 0, 100, None

ET_BETA_VARIANT = 0.7   # proposed change; 0.5 reproduces the released Dice loss
SEED = 0
BATCH_SIZE = 12         # as in the paper
""")

if PATCH is None:
    code(f"""
!nvidia-smi --query-gpu=name,memory.total --format=csv
%cd /content
!rm -rf multiPI_TransBTS && git clone -q -b {BRANCH} {REPO_URL} multiPI_TransBTS
%cd /content/multiPI_TransBTS
!pip install -q -r requirements-colab.txt
!mkdir -p networks/pretrained_pth && wget -q -nc -O networks/pretrained_pth/pvt_v2_b2.pth https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
!git log --oneline | head -5
""")
else:
    code("%%writefile /content/repro.patch\n" + PATCH)
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
%cd /content
!rm -rf multiPI_TransBTS && git clone -q https://github.com/JoetheReindeer/multiPI_TransBTS.git multiPI_TransBTS
%cd /content/multiPI_TransBTS
!echo >> /content/repro.patch
!git checkout -q 6ef9eee && git apply --whitespace=nowarn /content/repro.patch && git status --short | head -20
!pip install -q -r requirements-colab.txt
!mkdir -p networks/pretrained_pth && wget -q -nc -O networks/pretrained_pth/pvt_v2_b2.pth https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
""")

md("""
## 1. Checks that need no data
Patient overlap between the official train/val/test lists, what the released `hausdorff_95` returns for masks with a known distance, and the model's parameter count.
""")
code("!python -W ignore -m repro.audit --out results/audit.json")

md("""
## 2. Data
BraTS2020 training set from the public Kaggle mirror `awsaf49/brats20-dataset-training-validation` (about 4 GB).
If the anonymous download is refused, add your Kaggle API token as Colab secrets `KAGGLE_USERNAME` / `KAGGLE_KEY` (key icon in the left bar) and re-run.
""")
code("""
import kagglehub, os
RAW = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
print(RAW); print(os.listdir(RAW))
""")
code("""
sub = f"--n_subset {N_SUBSET} --n_heldout {N_HELDOUT}" if N_SUBSET else ""
!python -W ignore -m repro.prepare_data --raw "{RAW}" --out /content/prep {sub} --seed {SEED}
""")

md("""
## 3. Speed check
40 training iterations with the real batch size, to measure iterations per second and peak GPU memory before committing to a run.
If batch 12 does not fit in GPU memory, the batch size is reduced (this would be a deviation from the paper and is recorded in the results).
In the scaled preset the number of epochs is then set so that one run fits in `TRAIN_MINUTES_PER_RUN`.
""")
code("""
import json, subprocess, sys
for bs in (BATCH_SIZE, 8, 6, 4):
    cmd = f"{sys.executable} -W ignore -m repro.train --prep_dir /content/prep --out /content/bench {sub} --seed {SEED} --batch_size {bs} --epochs 1 --amp --max_train_iters 40 --no_eval"
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode == 0:
        BATCH_SIZE = bs
        break
    print(f"batch size {bs} failed:", r.stderr[-600:])
else:
    raise RuntimeError("the model does not fit on this GPU even at batch size 4")

b = json.load(open("/content/bench/results.json"))
ips = b["history"][-1]["it_per_s"]
print(f"batch size {BATCH_SIZE}: {ips:.2f} it/s on {b['device']}, peak GPU memory {b['peak_gpu_mem_gb']:.1f} GB, {b['iters_per_epoch']} iterations per epoch")
print(f"the paper's full setting (1628 iterations x 100 epochs at batch 12) would take about {1628 * 100 * 12 / BATCH_SIZE / ips / 3600:.0f} GPU-hours on this GPU")
if TRAIN_MINUTES_PER_RUN:
    fit = int(TRAIN_MINUTES_PER_RUN * 60 * ips / b["iters_per_epoch"])
    EPOCHS = max(3, min(EPOCHS, fit))
print(f"epochs per run: {EPOCHS}  (about {b['iters_per_epoch'] * EPOCHS / ips / 60:.0f} min of training per run)")
""")

md("## 4. Baseline: released model, released settings")
code("""
!python -W ignore -m repro.train --prep_dir /content/prep --out results/baseline --ckpt_dir /content/ckpt_baseline {sub} --seed {SEED} --batch_size {BATCH_SIZE} --epochs {EPOCHS} --amp --et_beta 0.5
""")

md("""
## 5. Proposed variant: ET loss with false-negative weight beta
Everything is identical to the baseline (same split, seed, schedule) except `--et_beta`.
""")
code("""
VAR = f"results/variant_et_beta{ET_BETA_VARIANT}"
!python -W ignore -m repro.train --prep_dir /content/prep --out {VAR} --ckpt_dir /content/ckpt_variant {sub} --seed {SEED} --batch_size {BATCH_SIZE} --epochs {EPOCHS} --amp --et_beta {ET_BETA_VARIANT}
""")

md("## 6. Report")
code("""
!python -W ignore -m repro.make_report --baseline results/baseline --variant {VAR} --out results > /dev/null
from IPython.display import Markdown, Image, display
display(Markdown(open("results/summary.md").read()))
for f in ("dice_protocols", "et_variant", "training_loss", "examples"):
    if os.path.exists(f"results/figures/{f}.png"):
        display(Image(f"results/figures/{f}.png", width=760))
""")
code("""
# compact machine-readable dump of the numbers (for copying into the README / slides)
import json
for run in ("results/baseline", VAR):
    r = json.load(open(f"{run}/results.json"))
    keep = {k: r[k] for k in ("split", "test", "unseen", "device", "peak_gpu_mem_gb", "iters_per_epoch", "params_million") if k in r}
    keep["history"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items()} for h in r["history"]]
    keep["config"] = {k: r["config"][k] for k in ("epochs", "batch_size", "n_subset", "n_heldout", "seed", "et_beta", "lr")}
    print("RESULT_JSON", run, json.dumps(keep, separators=(",", ":")))
""")
code("""
# download everything (results.json files, summary, figures, example arrays)
!cd /content/multiPI_TransBTS && zip -q -r /content/results.zip results
from google.colab import files
files.download("/content/results.zip")
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, len(cells), "cells")
