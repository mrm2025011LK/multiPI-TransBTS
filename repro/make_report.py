"""Turn results/*/results.json into tables (results/summary.md) and figures.

    python -m repro.make_report --baseline results/baseline --variant results/variant_et_beta0.7
"""
import argparse
import json
import os

import numpy as np

# "Ours" rows of Tables 2 and 3 of the paper (arXiv:2409.12167)
PAPER = {
    "2019": {"dice": {"WT": 0.953, "TC": 0.944, "ET": 0.904, "Mean": 0.934},
             "hd95": {"WT": 2.177, "TC": 1.538, "ET": 1.334, "Mean": 1.683},
             "sens": {"WT": 0.962, "TC": 0.954, "ET": 0.899, "Mean": 0.938}},
    "2020": {"dice": {"WT": 0.954, "TC": 0.947, "ET": 0.909, "Mean": 0.937},
             "hd95": {"WT": 2.167, "TC": 1.476, "ET": 1.271, "Mean": 1.638},
             "sens": {"WT": 0.961, "TC": 0.951, "ET": 0.909, "Mean": 0.940}},
}
REGIONS = ("WT", "TC", "ET")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.grid": True,
        "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })
    return plt


def _bars(ax, groups, series, ylim=None, fmt="{:.3f}"):
    """groups: list of labels; series: list of (name, color, values)."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, color, vals) in enumerate(series):
        pos = x + (i - (n - 1) / 2) * width
        ax.bar(pos, vals, width * 0.92, color=color, label=name, linewidth=0)
        for p, v in zip(pos, vals):
            ax.text(p, v + (ylim[1] - ylim[0]) * 0.012 if ylim else v, fmt.format(v),
                    ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.tick_params(length=0)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=n, fontsize=10.5)


def hausdorff_figure(audit_path, fig_dir):
    """Line chart of the Hausdorff check in results/audit.json (needs no training run)."""
    plt = _style()
    os.makedirs(fig_dir, exist_ok=True)
    with open(audit_path) as f:
        h = json.load(f)["hausdorff"]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    xs = [r["shift_px"] for r in h]
    ax.plot(xs, [r["spatial_hd95_px"] for r in h], color=ORANGE, linewidth=2, marker="o", markersize=6,
            label="Spatial HD95")
    ax.plot(xs, [r["released_hausdorff_95"] for r in h], color=BLUE, linewidth=2, marker="o", markersize=6,
            label="Released hausdorff_95")
    ax.text(xs[-1], h[-1]["spatial_hd95_px"], f"  {h[-1]['spatial_hd95_px']:.0f}", va="center", fontsize=10)
    ax.text(xs[-1], h[-1]["released_hausdorff_95"], f"  {h[-1]['released_hausdorff_95']:.1f}", va="center",
            fontsize=10)
    ax.set_xlabel("How far the predicted disc is shifted from the true one (pixels)")
    ax.set_ylabel("Reported distance")
    ax.set_title("A 20-pixel-radius disc moved sideways: what each metric reports")
    ax.legend(loc="upper left")
    ax.set_xlim(-3, xs[-1] + 9)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "hausdorff_check.png"), dpi=180)
    plt.close(fig)


def load(run_dir):
    with open(os.path.join(run_dir, "results.json")) as f:
        return json.load(f)


def fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def table_official(res, key):
    rows = ["| Region | Dice | Sensitivity | Precision | released `hausdorff_95` | spatial HD95 (px) |",
            "|---|---|---|---|---|---|"]
    for r in REGIONS + ("Mean",):
        m = res[key][r]
        rows.append(f"| {r} | {fmt(m['dice_pooled'])} | {fmt(m['sensitivity_pooled'])} | "
                    f"{fmt(m['precision_pooled'])} | {fmt(m['hd95_official'])} | {fmt(m['hd95_true_mean_px'], 2)} |")
    return "\n".join(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--variant", default=None)
    p.add_argument("--audit", default="results/audit.json")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    plt = _style()
    fig_dir = os.path.join(args.out, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    base = load(args.baseline)
    var = load(args.variant) if args.variant and os.path.exists(os.path.join(args.variant, "results.json")) else None
    cfg = base["config"]
    md = ["# Results", ""]
    md.append(f"Run: {base['device']}, {cfg['epochs']} epochs, batch {cfg['batch_size']}, "
              f"subset of {cfg['n_subset'] or 'all'} patients "
              f"({base['split']['train']['slices']} training slices), "
              f"{base['split']['unseen']['patients']} unseen patients, seed {cfg['seed']}. "
              f"Training time {sum(h['epoch_seconds'] for h in base['history']) / 60:.0f} min, "
              f"peak GPU memory {fmt(base['peak_gpu_mem_gb'], 1)} GB.")
    md.append("")

    # ---- 1. reproduction: paper vs official protocol vs unseen patients ----
    md += ["## Baseline (released settings)", "",
           f"### Official protocol: official test slices of the subset "
           f"({base['test']['n_slices']} slices; {base['split']['test']['patients_also_in_train']} of "
           f"{base['split']['test']['patients']} test patients also occur in training)", "",
           table_official(base, "test"), ""]
    if "unseen" in base:
        md += [f"### Same weights on unseen patients ({base['unseen']['n_patients']} patients, "
               f"{base['unseen']['n_slices']} slices, none in training)", "",
               table_official(base, "unseen"), "",
               "Per-patient Dice on unseen patients (mean ± std over patients): " +
               ", ".join(f"{r} {base['unseen'][r]['dice_per_patient_mean']:.3f} ± "
                         f"{base['unseen'][r]['dice_per_patient_std']:.3f}" for r in REGIONS), ""]

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    series = [("Paper, BraTS2020 (full run)", MUTED, [PAPER["2020"]["dice"][r] for r in REGIONS]),
              ("Ours, official test slices", BLUE, [base["test"][r]["dice_pooled"] for r in REGIONS])]
    if "unseen" in base:
        series.append(("Ours, unseen patients", ORANGE, [base["unseen"][r]["dice_pooled"] for r in REGIONS]))
    lo = min(min(s[2]) for s in series)
    _bars(ax, list(REGIONS), series, ylim=(max(0.0, np.floor((lo - 0.08) * 10) / 10), 1.04))
    ax.set_title("Dice by tumour region")
    ax.set_ylabel("Dice (pooled over test pixels)")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dice_protocols.png"), dpi=180)
    plt.close(fig)

    # ---- 2. variant ----
    if var is not None:
        key = "unseen" if "unseen" in base else "test"
        md += [f"## Proposed change: ET loss with beta = {var['config']['et_beta']} "
               f"(baseline beta = {cfg['et_beta']})", "",
               f"Evaluated on the {key} set, same split, same seed.", "",
               "| Model | Region | Dice | Sensitivity | Precision | per-patient Dice | spatial HD95 (px) |",
               "|---|---|---|---|---|---|---|"]
        for name, res in (("baseline", base), ("variant", var)):
            for r in REGIONS:
                m = res[key][r]
                md.append(f"| {name} | {r} | {fmt(m['dice_pooled'])} | {fmt(m['sensitivity_pooled'])} | "
                          f"{fmt(m['precision_pooled'])} | {fmt(m['dice_per_patient_mean'])} | "
                          f"{fmt(m['hd95_true_mean_px'], 2)} |")
        md.append("")
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        labels = ["Sensitivity", "Precision", "Dice"]
        ks = ["sensitivity_pooled", "precision_pooled", "dice_pooled"]
        series = [(f"Baseline (beta {cfg['et_beta']})", BLUE, [base[key]["ET"][k] for k in ks]),
                  (f"Variant (beta {var['config']['et_beta']})", ORANGE, [var[key]["ET"][k] for k in ks])]
        lo = min(min(s[2]) for s in series)
        _bars(ax, labels, series, ylim=(max(0.0, np.floor((lo - 0.08) * 10) / 10), 1.04))
        ax.set_title(f"Enhancing tumour on {'unseen patients' if key == 'unseen' else 'the test set'}")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "et_variant.png"), dpi=180)
        plt.close(fig)

    # ---- 3. training curves ----
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for name, res, color in (("Baseline", base, BLUE), ("Variant", var, ORANGE)):
        if res is None:
            continue
        h = res["history"]
        ax.plot([e["epoch"] for e in h], [e["train_loss"] for e in h], color=color, linewidth=2,
                marker="o", markersize=5, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss per epoch")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "training_loss.png"), dpi=180)
    plt.close(fig)

    # ---- 4. Hausdorff demonstration ----
    if os.path.exists(args.audit):
        hausdorff_figure(args.audit, fig_dir)

    # ---- 5. qualitative examples ----
    ex_name = "examples_unseen.npz" if os.path.exists(os.path.join(args.baseline, "examples_unseen.npz")) \
        else "examples_test.npz"
    ex_path = os.path.join(args.baseline, ex_name)
    if os.path.exists(ex_path):
        b = np.load(ex_path)
        v = np.load(os.path.join(args.variant, ex_name)) if var is not None and \
            os.path.exists(os.path.join(args.variant, ex_name)) else None
        n = min(4, len(b["names"]))
        cols = ["FLAIR", "T1ce", "Ground truth", "Baseline"] + (["Variant"] if v is not None else [])
        fig, axes = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 2.6 * n), squeeze=False)

        def overlay(t1ce, m):
            rgb = np.repeat(np.clip(t1ce, 0, 1)[..., None], 3, axis=2) * 0.85
            for k, c in ((0, (0.16, 0.47, 0.84)), (1, (0.92, 0.41, 0.20)), (2, (0.93, 0.63, 0.0))):
                rgb[m[k]] = 0.45 * rgb[m[k]] + 0.55 * np.array(c)
            return rgb

        for i in range(n):
            img = b["img"][i].astype(np.float32)
            panels = [np.repeat(img[0][..., None], 3, 2), np.repeat(img[2][..., None], 3, 2),
                      overlay(img[2], b["gt"][i]), overlay(img[2], b["pred"][i])]
            if v is not None:
                panels.append(overlay(img[2], v["pred"][i]))
            for j, pan in enumerate(panels):
                axes[i, j].imshow(np.clip(pan, 0, 1))
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].grid(False)
                for s in axes[i, j].spines.values():
                    s.set_visible(False)
                if i == 0:
                    axes[i, j].set_title(cols[j], fontsize=11, loc="center")
        fig.text(0.5, 0.005, "Overlays: blue = whole tumour, orange = tumour core, yellow = enhancing tumour",
                 ha="center", fontsize=10, color=MUTED)
        fig.tight_layout(rect=(0, 0.025, 1, 1))
        fig.savefig(os.path.join(fig_dir, "examples.png"), dpi=150)
        plt.close(fig)

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
