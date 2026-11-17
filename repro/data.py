"""Data preparation and splits for the multiPI-TransBTS reproduction.

The preprocessing follows data2020/dataset_preprocess.py from the official repo
step for step (SimpleITK Int16 read, percentile clipping on the brain mask,
division by the maximum, 224x224 centre crop, only slices that contain tumour,
channel order FLAIR / T1 / T1ce / T2). The only differences are practical:
one compressed file per patient instead of one .npy per slice, and float16
storage instead of float64, so that a subset fits in Colab RAM.
"""
import glob
import os

import numpy as np

MODALITIES = ("flair", "t1", "t1ce", "t2")  # channel order used by the official code
CROP = 224


# ----------------------------------------------------------------------------
# official preprocessing (copied in behaviour from data2020/dataset_preprocess.py)
# ----------------------------------------------------------------------------
def normalize(image, percentile_lower=0.2, percentile_upper=99.8):
    mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    res = res / res.max()
    return res


def crop_center(vol, croph=CROP, cropw=CROP):
    height, width = vol[0].shape
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return vol[:, starth:starth + croph, startw:startw + cropw]


def _find_file(case_dir, case, suffix):
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(case_dir, f"{case}_{suffix}{ext}")
        if os.path.exists(p):
            return p
    if suffix == "seg":  # the Kaggle mirror ships one mis-named label file (case 355)
        cand = [f for f in glob.glob(os.path.join(case_dir, "*"))
                if "seg" in os.path.basename(f).lower()]
        if cand:
            return cand[0]
    raise FileNotFoundError(f"{case}: no {suffix} volume in {case_dir}")


def find_cases(raw_root):
    """Return {case_name: case_dir} for every BraTS case folder below raw_root."""
    cases = {}
    for dirpath, _dirnames, filenames in os.walk(raw_root):
        if any(f.endswith(("_flair.nii.gz", "_flair.nii")) for f in filenames):
            name = os.path.basename(dirpath)
            if any("seg" in f.lower() for f in filenames):  # labelled cases only
                cases[name] = dirpath
    return dict(sorted(cases.items()))


def preprocess_case(case, case_dir, out_dir):
    """Preprocess one patient; writes <out_dir>/<case>.npz and returns the slice count."""
    import SimpleITK as sitk

    out_path = os.path.join(out_dir, f"{case}.npz")
    if os.path.exists(out_path):
        with np.load(out_path) as z:
            return int(z["slices"].shape[0])

    vols = []
    for mod in MODALITIES:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(_find_file(case_dir, case, mod), sitk.sitkInt16))
        vols.append(crop_center(normalize(arr)))
    seg = sitk.GetArrayFromImage(sitk.ReadImage(_find_file(case_dir, case, "seg"), sitk.sitkUInt8))
    seg = crop_center(seg)

    keep = [z for z in range(seg.shape[0]) if seg[z].max() != 0]
    img = np.stack([v[keep] for v in vols], axis=1).astype(np.float16)  # (n, 4, 224, 224)
    np.savez_compressed(out_path, img=img, mask=seg[keep].astype(np.uint8),
                        slices=np.asarray(keep, dtype=np.int16))
    return len(keep)


# ----------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------
def read_name_list(path):
    """The official lists mix \\n and bare \\r line endings."""
    with open(path, "r", newline="") as f:
        text = f.read()
    return [x.strip() for x in text.replace("\r", "\n").split("\n") if x.strip()]


def patient_of(slice_name):
    return slice_name.rsplit("_", 1)[0]


def slice_of(slice_name):
    return int(slice_name.rsplit("_", 1)[1])


def official_lists(repo_root, year="2020"):
    d = os.path.join(repo_root, f"data{year}")
    val_file = "val.txt" if year == "2020" else "val_list.txt"
    return {
        "train": read_name_list(os.path.join(d, "train_list.txt")),
        "val": read_name_list(os.path.join(d, val_file)),
        "test": read_name_list(os.path.join(d, "test_list.txt")),
    }


def choose_patients(all_patients, n_subset, n_heldout, seed=0):
    """Deterministic choice of the experiment subset S and of unseen patients H.

    S is the pool the official (slice-level) lists are restricted to.
    H is disjoint from S and is never used for training in any protocol.
    """
    all_patients = sorted(all_patients)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(all_patients))
    if n_subset is None or n_subset + n_heldout >= len(all_patients):
        # full run: nothing is left over for an unseen-patient set
        return all_patients, []
    subset = sorted(all_patients[i] for i in perm[:n_subset])
    heldout = sorted(all_patients[i] for i in perm[n_subset:n_subset + n_heldout])
    return subset, heldout


def build_split(repo_root, subset, heldout, protocol="official", seed=0, year="2020"):
    """Return {split_name: [slice names]}.

    protocol="official": the official train/val/test slice lists, restricted to
        the patients in `subset`. Slices of one patient land in all three splits.
    protocol="patient":  the patients in `subset` are divided 8:1:1 and every
        slice follows its patient.
    In both cases "unseen" holds all official slices of the `heldout` patients.
    """
    lists = official_lists(repo_root, year)
    every = lists["train"] + lists["val"] + lists["test"]
    sub, held = set(subset), set(heldout)
    split = {"unseen": sorted(n for n in every if patient_of(n) in held)}
    if protocol == "official":
        for k in ("train", "val", "test"):
            split[k] = [n for n in lists[k] if patient_of(n) in sub]
    elif protocol == "patient":
        pts = sorted(sub)
        rng = np.random.RandomState(seed + 1)
        rng.shuffle(pts)
        n_val = max(1, round(0.1 * len(pts)))
        n_test = max(1, round(0.1 * len(pts)))
        groups = {"val": set(pts[:n_val]), "test": set(pts[n_val:n_val + n_test]),
                  "train": set(pts[n_val + n_test:])}
        for k, g in groups.items():
            split[k] = sorted(n for n in every if patient_of(n) in g)
    else:
        raise ValueError(protocol)
    return split


def leakage_report(split):
    """How many patients of each evaluation split also occur in the training split."""
    train_p = {patient_of(n) for n in split["train"]}
    rep = {}
    for k in ("val", "test", "unseen"):
        p = {patient_of(n) for n in split.get(k, [])}
        rep[k] = {"slices": len(split.get(k, [])), "patients": len(p),
                  "patients_also_in_train": len(p & train_p)}
    rep["train"] = {"slices": len(split["train"]), "patients": len(train_p)}
    return rep


# ----------------------------------------------------------------------------
# in-memory dataset
# ----------------------------------------------------------------------------
class SliceStore:
    """Loads the preprocessed patients needed for a split into RAM once."""

    def __init__(self, prep_dir, patients):
        self.index = {}
        imgs, masks = [], []
        offset = 0
        for p in sorted(patients):
            with np.load(os.path.join(prep_dir, f"{p}.npz")) as z:
                img, mask, slices = z["img"], z["mask"], z["slices"]
            imgs.append(img)
            masks.append(mask)
            for i, s in enumerate(slices):
                self.index[f"{p}_{int(s)}"] = offset + i
            offset += len(slices)
        self.img = np.concatenate(imgs, axis=0)    # (N, 4, 224, 224) float16
        self.mask = np.concatenate(masks, axis=0)  # (N, 224, 224) uint8, labels 0/1/2/4

    def rows(self, names):
        missing = [n for n in names if n not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} slices from the split list were not produced by "
                           f"preprocessing, e.g. {missing[:3]}")
        return np.asarray([self.index[n] for n in names], dtype=np.int64)


def region_targets(mask):
    """BraTS labels -> (WT, TC, ET) binary maps, as in data2020/dataset_example.py.

    mask: integer tensor (B, H, W) with labels 0, 1 (NCR/NET), 2 (ED), 4 (ET).
    """
    import torch
    wt = mask > 0
    tc = (mask == 1) | (mask == 4)
    et = mask == 4
    return torch.stack([wt, tc, et], dim=1).float()
