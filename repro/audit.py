"""Checks of the released code that need no dataset and no GPU.

    python -m repro.audit            # prints the findings and writes results/audit.json

1. Split: are the official train / val / test lists disjoint at patient level?
2. Hausdorff: what does kits.metrics.hausdorff_95 return for masks whose true
   distance is known?
3. Model: parameter count, and parameters that never receive a gradient.
"""
import argparse
import json
import os

import numpy as np

from repro import data as D
from repro.metrics import hd95


def split_audit(repo_root, year):
    lists = D.official_lists(repo_root, year)
    pats = {k: {D.patient_of(n) for n in v} for k, v in lists.items()}
    n_all = sum(len(v) for v in lists.values())
    return {
        "slices": {k: len(v) for k, v in lists.items()},
        "slice_fraction": {k: round(len(v) / n_all, 4) for k, v in lists.items()},
        "patients_total": len(set().union(*pats.values())),
        "patients": {k: len(v) for k, v in pats.items()},
        "slice_overlap_train_test": len(set(lists["train"]) & set(lists["test"])),
        "test_patients_also_in_train": len(pats["test"] & pats["train"]),
        "val_patients_also_in_train": len(pats["val"] & pats["train"]),
        "test_patients_also_in_val": len(pats["test"] & pats["val"]),
    }


def neighbour_audit(repo_root, year):
    """For each test slice: is the slice directly above or below it (same patient) in train?"""
    lists = D.official_lists(repo_root, year)
    train = set(lists["train"])
    hits = 0
    for n in lists["test"]:
        p, s = D.patient_of(n), D.slice_of(n)
        if f"{p}_{s - 1}" in train or f"{p}_{s + 1}" in train:
            hits += 1
    return {"test_slices": len(lists["test"]), "with_adjacent_slice_in_train": hits,
            "fraction": round(hits / len(lists["test"]), 4)}


def hausdorff_audit():
    from kits import metrics as official
    yy, xx = np.mgrid[:224, :224]

    def disc(cx, cy=112, r=20):
        return ((yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2).astype(np.int64)

    ref = disc(60)
    rows = []
    for shift in (0, 5, 30, 60, 90):
        moved = disc(60 + shift)
        rows.append({
            "shift_px": shift,
            "released_hausdorff_95": round(float(official.hausdorff_95(moved[None], ref[None])), 3),
            "spatial_hd95_px": round(hd95(moved, ref), 3),
        })
    return rows


def model_audit(pretrained):
    import torch
    from networks.pvt_new import multiPI_TransBTS
    model = multiPI_TransBTS([0.5, 0.5], pretrained_path=pretrained)
    out = model(torch.rand(1, 4, 224, 224))
    sum(o.mean() for o in out[1:]).backward()
    total = sum(p.numel() for p in model.parameters())
    unused = {}
    for k, p in model.named_parameters():
        if p.grad is None:
            top = ".".join(k.split(".")[:2]) if k.startswith("backbone0") else k.split(".")[0]
            unused[top] = unused.get(top, 0) + p.numel()
    return {"params_million": round(total / 1e6, 2),
            "never_used_million": round(sum(unused.values()) / 1e6, 2),
            "never_used_modules": {k: round(v / 1e6, 2) for k, v in sorted(unused.items())}}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default=".")
    p.add_argument("--pretrained", default="./networks/pretrained_pth/pvt_v2_b2.pth")
    p.add_argument("--out", default="results/audit.json")
    args = p.parse_args(argv)
    res = {"split": {}, "adjacent": {}}
    for year in ("2019", "2020"):
        res["split"][year] = split_audit(args.repo_root, year)
        res["adjacent"][year] = neighbour_audit(args.repo_root, year)
    res["hausdorff"] = hausdorff_audit()
    if os.path.exists(args.pretrained):
        res["model"] = model_audit(args.pretrained)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    try:
        from repro.make_report import hausdorff_figure
        hausdorff_figure(args.out, os.path.join(os.path.dirname(args.out) or ".", "figures"))
    except ImportError:  # matplotlib missing: the JSON is still complete
        pass


if __name__ == "__main__":
    main()
