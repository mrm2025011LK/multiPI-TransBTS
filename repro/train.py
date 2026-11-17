"""Train and evaluate multiPI-TransBTS with the released model code.

Defaults are the released settings (train.py / trainer.py of the official repo):
SGD, lr 0.01, momentum 0.9, weight decay 1e-4, poly schedule, batch 12,
224x224 slices, ImageNet-pretrained PVTv2-B2 branches, no augmentation, the
final-epoch weights are the ones evaluated.

Example (scaled Colab run):
    python -m repro.train --prep_dir /content/prep --out results/baseline \
        --n_subset 80 --n_heldout 30 --epochs 15 --amp
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from repro import data as D
from repro.losses import official_loss
from repro.metrics import Evaluator


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prep_dir", required=True, help="output folder of repro.prepare_data")
    p.add_argument("--repo_root", default=".", help="repo root (for the official split lists)")
    p.add_argument("--out", required=True, help="folder for results.json, log and examples")
    p.add_argument("--ckpt_dir", default=None, help="where to keep the checkpoint (default: --out)")
    p.add_argument("--pretrained", default="./networks/pretrained_pth/pvt_v2_b2.pth")
    # data / protocol
    p.add_argument("--protocol", default="official", choices=["official", "patient"])
    p.add_argument("--n_subset", type=int, default=None,
                   help="patients in the experiment subset (None = all 369, the full run)")
    p.add_argument("--n_heldout", type=int, default=0,
                   help="extra patients kept entirely out of training, for an unseen-patient test")
    p.add_argument("--seed", type=int, default=0)
    # released hyper-parameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--fe_ratio", type=float, default=0.5, help="K of the feature-enhancement module")
    # the one changed parameter of the proposed variant
    p.add_argument("--et_beta", type=float, default=0.5,
                   help="false-negative weight of the ET overlap loss; 0.5 = released Dice loss")
    # practicalities
    p.add_argument("--amp", action="store_true", help="mixed precision (needed to fit a 16 GB GPU)")
    p.add_argument("--val_every", type=int, default=5)
    p.add_argument("--max_train_iters", type=int, default=None, help="stop early (speed benchmark)")
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--no_official_hd", action="store_true")
    p.add_argument("--n_examples", type=int, default=6)
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, store, names, device, amp, batch_size=12, official_hd=True, keep_examples=0):
    model.eval()
    ev = Evaluator(with_official_hd=official_hd)
    rows = store.rows(names)
    examples = []
    want = set(np.linspace(0, len(names) - 1, keep_examples).astype(int).tolist()) if keep_examples else set()
    for start in range(0, len(rows), batch_size):
        idx = rows[start:start + batch_size]
        img = torch.from_numpy(store.img[idx].astype(np.float32)).to(device)
        mask = torch.from_numpy(store.mask[idx].astype(np.int64)).to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(img)[0]
        pred = (torch.sigmoid(logits.float()) > 0.5).cpu().numpy()
        gt = D.region_targets(mask).bool().cpu().numpy()
        batch_names = names[start:start + batch_size]
        ev.update(pred, gt, [D.patient_of(n) for n in batch_names])
        for j, n in enumerate(batch_names):
            if start + j in want:
                examples.append({"name": n, "img": store.img[idx[j]], "gt": gt[j], "pred": pred[j]})
    return ev.summary(), examples


def main(argv=None):
    args = get_args(argv)
    os.makedirs(args.out, exist_ok=True)
    ckpt_dir = args.ckpt_dir or args.out
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data ----------------
    available = sorted(os.path.basename(f)[:-4] for f in os.listdir(args.prep_dir) if f.endswith(".npz"))
    lists = D.official_lists(args.repo_root)
    listed = sorted({D.patient_of(n) for k in lists for n in lists[k]})
    if args.n_subset is None:
        subset, heldout = listed, []
    else:
        subset, heldout = D.choose_patients(listed, args.n_subset, args.n_heldout, args.seed)
    missing = [p for p in subset + heldout if p not in available]
    if missing:
        raise SystemExit(f"{len(missing)} required patients are not preprocessed, e.g. {missing[:3]}")
    split = D.build_split(args.repo_root, subset, heldout, args.protocol, args.seed)
    leak = D.leakage_report(split)
    print("split:", json.dumps(leak))
    store = D.SliceStore(args.prep_dir, subset + heldout)
    print(f"loaded {store.img.shape[0]} slices, {store.img.nbytes / 1e9:.2f} GB in RAM")
    # a listed slice that preprocessing did not produce is dropped and reported, not fatal
    not_produced = {k: sum(n not in store.index for n in v) for k, v in split.items()}
    if any(not_produced.values()):
        print("WARNING: listed slices missing from the preprocessed data:", not_produced)
        split = {k: [n for n in v if n in store.index] for k, v in split.items()}
        leak = D.leakage_report(split)

    # ---------------- model (released code) ----------------
    from networks.pvt_new import multiPI_TransBTS
    model = multiPI_TransBTS([args.fe_ratio, args.fe_ratio], pretrained_path=args.pretrained).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    train_rows = store.rows(split["train"])
    iters_per_epoch = len(train_rows) // args.batch_size  # drop_last=True, as released
    total_iters = iters_per_epoch * args.epochs
    history = []
    start_epoch = 0
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        history, start_epoch = ck["history"], ck["epoch"]
        print(f"resumed from epoch {start_epoch}")

    rng = np.random.RandomState(args.seed)
    for _ in range(start_epoch):  # keep the batch order identical after a resume
        rng.permutation(len(train_rows))

    # ---------------- training ----------------
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    done_iters = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        order = train_rows[rng.permutation(len(train_rows))]
        running, t0 = 0.0, time.time()
        for it in range(iters_per_epoch):
            T = epoch * iters_per_epoch + it
            lr = args.lr * (1 - T / total_iters) ** 0.9  # kits.schedulers 'poly'
            for g in optimizer.param_groups:
                g["lr"] = lr
            idx = np.sort(order[it * args.batch_size:(it + 1) * args.batch_size])
            img = torch.from_numpy(store.img[idx].astype(np.float32)).to(device, non_blocking=True)
            mask = torch.from_numpy(store.mask[idx].astype(np.int64)).to(device, non_blocking=True)
            target = D.region_targets(mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                _, out_wt, out_tc, out_et = model(img)
            loss, _parts = official_loss(out_wt, out_tc, out_et, target, et_beta=args.et_beta)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            done_iters += 1
            if it % 50 == 0:
                print(f"epoch {epoch + 1}/{args.epochs} it {it}/{iters_per_epoch} "
                      f"loss {running / (it + 1):.4f} lr {lr:.5f} "
                      f"{(it + 1) / (time.time() - t0):.2f} it/s", flush=True)
            if args.max_train_iters and done_iters >= args.max_train_iters:
                break
        rec = {"epoch": epoch + 1, "train_loss": running / (it + 1), "lr": lr,
               "epoch_seconds": time.time() - t0, "it_per_s": (it + 1) / (time.time() - t0)}
        if args.max_train_iters and done_iters >= args.max_train_iters:
            history.append(rec)
            break
        if not args.no_eval and ((epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs):
            val, _ = evaluate(model, store, split["val"], device, args.amp, args.batch_size, official_hd=False)
            rec["val"] = {r: val[r]["dice_pooled"] for r in ("WT", "TC", "ET")}
            print("val dice:", rec["val"], flush=True)
        history.append(rec)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "history": history, "epoch": epoch + 1}, ckpt_path)

    result = {
        "config": vars(args),
        "split": leak,
        "listed_slices_not_produced": not_produced,
        "history": history,
        "train_seconds_this_session": time.time() - t_train,
        "iters_per_epoch": iters_per_epoch,
        "params_million": sum(p.numel() for p in model.parameters()) / 1e6,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
        "torch": torch.__version__,
    }

    # ---------------- final evaluation (final-epoch weights, as released) ----------------
    if not args.no_eval:
        for name in ("test", "unseen"):
            if not split.get(name):
                continue
            t0 = time.time()
            summary, examples = evaluate(model, store, split[name], device, args.amp, args.batch_size,
                                         official_hd=not args.no_official_hd,
                                         keep_examples=args.n_examples if name == "unseen" or not split.get("unseen") else 0)
            result[name] = summary
            print(f"{name} ({time.time() - t0:.0f}s):", json.dumps(
                {r: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary[r].items()}
                 for r in ("WT", "TC", "ET")}), flush=True)
            if examples:
                np.savez_compressed(os.path.join(args.out, f"examples_{name}.npz"),
                                    names=np.array([e["name"] for e in examples]),
                                    img=np.stack([e["img"] for e in examples]),
                                    gt=np.stack([e["gt"] for e in examples]),
                                    pred=np.stack([e["pred"] for e in examples]))
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", os.path.join(args.out, "results.json"))
    return result


if __name__ == "__main__":
    main()
