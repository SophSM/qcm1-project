"""Train one segmentation model (GPU job, via slurm/train.slurm -- NOT on the login node).

    python src/train.py --task apo  --variant B --epochs 60 --name apo_best
    python src/train.py --task fasc --variant B --epochs 60 --name fasc_best

apo : cached crop at 512x512, whole-image training.
fasc: cached crop at height 768, random 512x512 patches (70 % centred on a fascicle pixel); validation by sliding
      window over the whole cached image.
Best weights (by validation Dice) -> models/<name>.pt ; history -> outputs/train_<name>.csv/.png
"""
import argparse
import json
import os
import random
import sys
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from geometry import dominant_angle, fascicle_segments  # noqa: E402
from model import DiceBCE, build_model, predict_sliding, predict_whole, to_tensor  # noqa: E402

PATCH = 512


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_input(task, variant, stem):
    v = "B" if variant == "A" else variant
    x = np.array(Image.open(config.OUTPUTS / "cache" / task / v / stem))
    if variant == "A":
        x = np.dstack([x[..., 0]] * 3)
    return x


def load_target(task, stem):
    return (np.array(Image.open(config.OUTPUTS / "cache" / task / "masks" / stem)) > 127).astype(np.float32)


def augment():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), rotate=(-10, 10), border_mode=cv2.BORDER_REFLECT_101,
                 p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ])


class SegDS(Dataset):
    def __init__(self, task, variant, stems, train):
        self.task, self.variant, self.stems, self.train = task, variant, stems, train
        self.aug = augment() if train else None

    def __len__(self):
        return len(self.stems)

    def _patch(self, x, m):
        H, W = m.shape
        if H < PATCH or W < PATCH:
            ph, pw = max(0, PATCH - H), max(0, PATCH - W)
            x = cv2.copyMakeBorder(x, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            m = cv2.copyMakeBorder(m, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            H, W = m.shape
        if random.random() < 0.7 and m.any():
            ys, xs = np.nonzero(m)
            k = random.randrange(len(ys))
            cy = ys[k] + random.randint(-PATCH // 3, PATCH // 3)
            cx = xs[k] + random.randint(-PATCH // 3, PATCH // 3)
            y0 = int(np.clip(cy - PATCH // 2, 0, H - PATCH))
            x0 = int(np.clip(cx - PATCH // 2, 0, W - PATCH))
        else:
            y0, x0 = random.randint(0, H - PATCH), random.randint(0, W - PATCH)
        return x[y0:y0 + PATCH, x0:x0 + PATCH], m[y0:y0 + PATCH, x0:x0 + PATCH]

    def __getitem__(self, i):
        s = self.stems[i]
        x, m = load_input(self.task, self.variant, s), load_target(self.task, s)
        if self.train:
            if self.task == "fasc":
                x, m = self._patch(x, m)
            elif random.random() < 0.3:
                # narrow scan window (e.g. Philips Lumify crops, ~0.65 width/height): take 40-100 % of the width
                # and stretch it back to 512, as the whole-crop resize does at test time
                H, W = m.shape
                w = int(W * random.uniform(0.4, 1.0))
                x0 = random.randint(0, W - w)
                x = cv2.resize(x[:, x0:x0 + w], (W, H), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(m[:, x0:x0 + w], (W, H), interpolation=cv2.INTER_NEAREST)
            r = self.aug(image=x, mask=m)
            x, m = r["image"], r["mask"]
        return to_tensor(x), torch.from_numpy(m[None].astype(np.float32))


def worker_init(_):
    """Distinct, reproducible random streams per worker (torch gives each worker its own initial seed)."""
    s = torch.initial_seed() % 2**31
    random.seed(s)
    np.random.seed(s)
    info = torch.utils.data.get_worker_info()
    if info is not None and info.dataset.aug is not None:
        info.dataset.aug.set_random_seed(s)


def dice_iou(p, t):
    inter = (p & t).sum()
    s = p.sum() + t.sum()
    union = (p | t).sum()
    if s == 0:
        return 1.0, 1.0
    return 2 * inter / s, inter / max(union, 1)


def pad32(x):
    _, H, W = x.shape
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    if ph or pw:
        x = torch.nn.functional.pad(x[None], (0, pw, 0, ph), mode="reflect")[0]
    return x, H, W


def validate(model, task, variant, stems, device, angles=False):
    model.eval()
    dices, ious, aerr, nmiss = [], [], [], 0
    for s in stems:
        x = to_tensor(load_input(task, variant, s))
        t = load_target(task, s) > 0.5
        if task == "apo":
            p = predict_whole(model, x, device, tta=False)
        else:
            x, H, W = pad32(x)
            p = predict_sliding(model, x, device, tta=False)[:H, :W]
        pb = p > 0.5
        d, j = dice_iou(pb, t)
        dices.append(d)
        ious.append(j)
        if angles and task == "fasc":
            at = dominant_angle(fascicle_segments(t, min_len=30))
            ap = dominant_angle(fascicle_segments(pb, min_len=30))
            if np.isfinite(at):
                if np.isfinite(ap):
                    aerr.append(abs(at - ap))
                else:
                    nmiss += 1
    model.train()
    res = dict(dice=float(np.mean(dices)), iou=float(np.mean(ious)))
    if angles and task == "fasc":
        res.update(angle_mae=float(np.mean(aerr)) if aerr else float("nan"),
                   angle_medae=float(np.median(aerr)) if aerr else float("nan"), angle_missing=nmiss)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["apo", "fasc"])
    ap.add_argument("--variant", default="B", choices=list("ABCD"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--name", default=None)
    ap.add_argument("--bs", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--max_batches", type=int, default=0, help="smoke test: limit batches per epoch")
    ap.add_argument("--all_data", action="store_true", help="also train on the validation split")
    ap.add_argument("--save_last", action="store_true",
                    help="keep the last epoch instead of the best validation epoch (for --all_data runs)")
    args = ap.parse_args()
    name = args.name or f"{args.task}_{args.variant}"
    seed_all(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)
    print("config", json.dumps(vars(args)), flush=True)

    split = pd.read_csv(config.OUTPUTS / f"split_{args.task}.csv")
    split["stem"] = split.file.str.replace(r"\.[^.]+$", ".png", regex=True)
    tr = split[split.fold == "train"].stem.tolist() if not args.all_data else split.stem.tolist()
    va = split[split.fold == "val"].stem.tolist()
    if args.max_batches:
        va = va[:20]
    print(f"train {len(tr)}  val {len(va)}", flush=True)

    dl = DataLoader(SegDS(args.task, args.variant, tr, True), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True, pin_memory=True, persistent_workers=True,
                    worker_init_fn=worker_init)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler()
    loss_fn = DiceBCE()
    os.makedirs(config.MODELS, exist_ok=True)
    hist, best, bad = [], -1, 0
    for ep in range(1, args.epochs + 1):
        t0, tl, nb = time.time(), 0.0, 0
        for b, (x, y) in enumerate(dl):
            if args.max_batches and b >= args.max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
            nb += 1
        sched.step()
        last = ep == args.epochs
        v = validate(model, args.task, args.variant, va, device, angles=(ep % 5 == 0 or last))
        row = dict(epoch=ep, train_loss=tl / max(nb, 1), lr=opt.param_groups[0]["lr"], sec=time.time() - t0, **v)
        hist.append(row)
        print(json.dumps({k: (round(v_, 4) if isinstance(v_, float) else v_) for k, v_ in row.items()}), flush=True)
        if args.save_last:
            if last:
                torch.save(dict(state_dict=model.state_dict(), task=args.task, variant=args.variant, epoch=ep,
                                val=v, args=vars(args)), config.MODELS / f"{name}.pt")
        elif v["dice"] > best:
            best, bad = v["dice"], 0
            torch.save(dict(state_dict=model.state_dict(), task=args.task, variant=args.variant, epoch=ep,
                            val=v, args=vars(args)), config.MODELS / f"{name}.pt")
        elif not args.save_last:
            bad += 1
        pd.DataFrame(hist).to_csv(config.OUTPUTS / f"train_{name}.csv", index=False)
        if bad >= args.patience:
            print("early stop at epoch", ep, flush=True)
            break
    # final: angle metrics of the best checkpoint
    ck = torch.load(config.MODELS / f"{name}.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    fin = validate(model, args.task, args.variant, va, device, angles=True)
    ck["val"] = fin
    torch.save(ck, config.MODELS / f"{name}.pt")
    print("BEST", json.dumps(dict(name=name, epoch=ck["epoch"], **fin)), flush=True)
    plot(hist, name)


def plot(hist, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(hist)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h.epoch, h.train_loss)
    ax[0].set_title("training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(h.epoch, h.dice, label="val Dice")
    ax[1].plot(h.epoch, h.iou, label="val IoU")
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_title(name)
    plt.tight_layout()
    os.makedirs(config.OUTPUTS / "figures", exist_ok=True)
    plt.savefig(config.OUTPUTS / "figures" / f"train_{name}.png", dpi=90)


if __name__ == "__main__":
    main()
