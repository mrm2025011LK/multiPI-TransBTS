"""Preprocess BraTS2020 training cases the way the official repo does.

    python -m repro.prepare_data --raw /content/brats20 --out /content/prep \
        --n_subset 80 --n_heldout 30

Only the patients needed for the chosen subset are processed. Afterwards the
slice names produced here are compared with the official split lists: if the
preprocessing matches the released one, every listed slice of every processed
patient must exist, and nothing extra.
"""
import argparse
import json
import os
import time

import numpy as np

from repro import data as D


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="folder that contains the BraTS20_Training_XXX folders")
    p.add_argument("--out", required=True)
    p.add_argument("--repo_root", default=".")
    p.add_argument("--n_subset", type=int, default=None)
    p.add_argument("--n_heldout", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    lists = D.official_lists(args.repo_root)
    listed_slices = {}
    for k in lists:
        for n in lists[k]:
            listed_slices.setdefault(D.patient_of(n), set()).add(D.slice_of(n))
    listed = sorted(listed_slices)
    if args.n_subset is None:
        wanted = listed
    else:
        subset, heldout = D.choose_patients(listed, args.n_subset, args.n_heldout, args.seed)
        wanted = subset + heldout

    cases = D.find_cases(args.raw)
    print(f"{len(cases)} labelled cases found under {args.raw}; processing {len(wanted)}")
    t0 = time.time()
    n_slices, mismatched = 0, []
    for i, case in enumerate(wanted):
        if case not in cases:
            raise SystemExit(f"{case} is in the official lists but not in {args.raw}")
        n = D.preprocess_case(case, cases[case], args.out)
        n_slices += n
        with np.load(os.path.join(args.out, f"{case}.npz")) as z:
            produced = set(int(s) for s in z["slices"])
        if produced != listed_slices[case]:
            mismatched.append({"case": case, "produced": len(produced), "listed": len(listed_slices[case]),
                               "only_produced": sorted(produced - listed_slices[case])[:5],
                               "only_listed": sorted(listed_slices[case] - produced)[:5]})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(wanted)} patients, {n_slices} slices, {time.time() - t0:.0f}s", flush=True)
    report = {"patients": len(wanted), "slices": n_slices,
              "patients_matching_official_lists": len(wanted) - len(mismatched),
              "mismatched": mismatched}
    with open(os.path.join(args.out, "preprocess_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "mismatched"}))
    if mismatched:
        print("WARNING: slice sets differ from the official lists for", len(mismatched), "patients:", mismatched[:3])


if __name__ == "__main__":
    main()
