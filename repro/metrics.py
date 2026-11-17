"""Two sets of metrics.

1. "official": exactly what the released trainer.py computes, by calling the
   released kits.metrics code (pooled Dice / Sensitivity from one confusion
   matrix, and the released `hausdorff_95`).
2. "standard": metrics as they are usually defined for BraTS-style reporting
   (per-patient Dice, precision, and a spatial 95th-percentile Hausdorff
   distance between the two mask surfaces).

Why both: `kits.metrics.hausdorff_95` passes whole 2D masks to
`hausdorff.hausdorff_distance`, which expects an (n_points, n_dims) array. Each
image ROW is therefore treated as one 224-dimensional point, and the result is
multiplied by 0.95. That number is not a distance in the image plane and is not
a 95th percentile (see repro/audit.py for a demonstration).
"""
from collections import defaultdict

import numpy as np
from scipy import ndimage

REGIONS = ("WT", "TC", "ET")


# ----------------------------------------------------------------------------
# standard definitions
# ----------------------------------------------------------------------------
def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def hd95(pred, gt, spacing=1.0):
    """Symmetric 95th-percentile Hausdorff distance between mask surfaces (2D or 3D).

    Returns nan when either mask is empty (the distance is undefined there).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp, sg = _surface(pred), _surface(gt)
    d_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([d_to_g[sp], d_to_p[sg]]), 95))


def _dice(tp, fp, fn):
    den = 2 * tp + fp + fn
    return 1.0 if den == 0 else 2 * tp / den  # both empty -> 1 (BraTS convention)


class Evaluator:
    """Accumulates predictions slice by slice and reports both metric sets."""

    def __init__(self, with_official_hd=True):
        self.with_official_hd = with_official_hd
        self.reset()

    def reset(self):
        self.pooled = np.zeros((3, 4), dtype=np.float64)  # tp, fp, fn, tn per region
        self.per_patient = defaultdict(lambda: np.zeros((3, 3), dtype=np.float64))
        self.hd_true = [[] for _ in REGIONS]              # per slice, both masks non-empty
        self.hd_undefined = np.zeros(3, dtype=np.int64)   # exactly one of the masks empty
        self.hd_official_batch = [[] for _ in REGIONS]    # one value per batch, as upstream
        self.n_slices = 0

    def update(self, pred, gt, patients):
        """pred, gt: bool arrays (B, 3, H, W); patients: list of B patient ids."""
        pred = pred.astype(bool)
        gt = gt.astype(bool)
        self.n_slices += pred.shape[0]
        for r in range(3):
            p, g = pred[:, r], gt[:, r]
            tp = (p & g).sum(axis=(1, 2))
            fp = (p & ~g).sum(axis=(1, 2))
            fn = (~p & g).sum(axis=(1, 2))
            self.pooled[r] += [tp.sum(), fp.sum(), fn.sum(), p[0].size * len(p) - (tp + fp + fn).sum()]
            for i, pid in enumerate(patients):
                self.per_patient[pid][r] += [tp[i], fp[i], fn[i]]
                if p[i].any() and g[i].any():
                    self.hd_true[r].append(hd95(p[i], g[i]))
                elif p[i].any() != g[i].any():
                    self.hd_undefined[r] += 1
            if self.with_official_hd:
                from kits import metrics as official  # released code, unchanged
                self.hd_official_batch[r].append(
                    official.hausdorff_95(p.astype(np.int64), g.astype(np.int64)))

    def summary(self):
        out = {"n_slices": int(self.n_slices), "n_patients": len(self.per_patient)}
        for r, name in enumerate(REGIONS):
            tp, fp, fn, tn = self.pooled[r]
            per_pat = [_dice(*v[r]) for v in self.per_patient.values()]
            out[name] = {
                # --- as computed by the released trainer.py ---
                "dice_pooled": float(2 * tp / max(2 * tp + fp + fn, 1)),
                "sensitivity_pooled": float(tp / max(tp + fn, 1)),
                "specificity_pooled": float(tn / max(tn + fp, 1)),
                "hd95_official": (float(np.mean(self.hd_official_batch[r]))
                                  if self.hd_official_batch[r] else None),
                # --- standard definitions ---
                "precision_pooled": float(tp / max(tp + fp, 1)),
                "dice_per_patient_mean": float(np.mean(per_pat)),
                "dice_per_patient_std": float(np.std(per_pat)),
                "hd95_true_mean_px": (float(np.mean(self.hd_true[r])) if self.hd_true[r] else None),
                "hd95_true_median_px": (float(np.median(self.hd_true[r])) if self.hd_true[r] else None),
                "hd95_true_n_slices": len(self.hd_true[r]),
                "hd95_undefined_slices": int(self.hd_undefined[r]),
            }
        keys = ("dice_pooled", "sensitivity_pooled", "precision_pooled", "dice_per_patient_mean",
                "hd95_official", "hd95_true_mean_px")
        out["Mean"] = {}
        for k in keys:
            vals = [out[n][k] for n in REGIONS]
            out["Mean"][k] = None if any(v is None for v in vals) else float(np.mean(vals))
        return out
