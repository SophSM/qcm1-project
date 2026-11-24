"""Run both models on a set of images and save probability masks (uint8 PNG, full original image size).

    python src/predict.py --set test     --apo models/apo_best.pt --fasc models/fasc_best.pt --out test
    python src/predict.py --set apo_val  --apo models/apo_best.pt  --out val_<tag>     (apo validation images)
    python src/predict.py --set fasc_val --fasc models/fasc_best.pt --out val_<tag>    (fascicle validation images)

Everything is computed from the image content: crop box (crop.find_crop), preprocessing (variant stored in the
checkpoint). Output: outputs/pred_masks/<out>/{apo,fasc}/<stem>.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from crop import find_crop  # noqa: E402
from data import list_images  # noqa: E402
from model import load_checkpoint, predict_sliding, predict_whole, to_tensor  # noqa: E402
from preprocess import crop_resize_gray, make_input  # noqa: E402
from train import pad32  # noqa: E402


def image_list(which):
    if which == "test":
        d = config.TEST_DIR
        return [d / f for f in list_images(d)]
    task = which.split("_")[0]
    sp = pd.read_csv(config.OUTPUTS / f"split_{task}.csv")
    files = sp[sp.fold == "val"].file if which.endswith("_val") else sp.file
    return [config.DIRS[task]["imgs"] / f for f in files]


class PrepDS(Dataset):
    def __init__(self, paths, needs):
        self.paths, self.needs = paths, needs  # needs: {"apo": {"A", "B"}, "fasc": {"A"}}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        rgb = np.array(Image.open(p).convert("RGB"))
        g = np.array(Image.open(p).convert("L"))
        box = find_crop(rgb)
        out = dict(name=p.name, shape=np.array(g.shape), box=np.array(box))
        for task, vs in self.needs.items():
            gc = crop_resize_gray(g, box, task)
            for v in vs:
                out[f"{task}_{v}"] = to_tensor(make_input(gc, v))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--apo", nargs="*", default=[], help="one or more checkpoints (probabilities are averaged)")
    ap.add_argument("--fasc", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_tta", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}  # task -> list of (model, variant)
    for task in ("apo", "fasc"):
        for p in getattr(args, task):
            m, ck = load_checkpoint(p, device)
            models.setdefault(task, []).append((m, ck["variant"]))
            print(task, p, "variant", ck["variant"], "epoch", ck["epoch"], flush=True)
    needs = {t: {v for _, v in ms} for t, ms in models.items()}
    paths = image_list(args.set)
    outdir = config.OUTPUTS / "pred_masks" / args.out
    for t in models:
        os.makedirs(outdir / t, exist_ok=True)
    rows = []
    dl = DataLoader(PrepDS(paths, needs), batch_size=None, num_workers=args.workers)
    for i, b in enumerate(dl):
        H, W = b["shape"].tolist()
        x0, y0, x1, y1 = b["box"].tolist()
        stem = os.path.splitext(b["name"])[0] + ".png"
        for task, ms in models.items():
            probs = []
            for m, v in ms:
                x = b[f"{task}_{v}"]
                if task == "apo":
                    probs.append(predict_whole(m, x, device, tta=not args.no_tta))
                else:
                    xp, h, w = pad32(x)
                    probs.append(predict_sliding(m, xp, device, tta=not args.no_tta)[:h, :w])
            p = np.mean(probs, 0)
            p = cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            full = np.zeros((H, W), np.uint8)
            full[y0:y1, x0:x1] = np.clip(p * 255 + 0.5, 0, 255).astype(np.uint8)
            Image.fromarray(full).save(outdir / task / stem)
        rows.append(dict(file=b["name"], W=W, H=H, x0=x0, y0=y0, x1=x1, y1=y1))
        if i % 50 == 0:
            print(i, len(paths), flush=True)
    pd.DataFrame(rows).to_csv(outdir / f"boxes_{args.set}.csv", index=False)
    if args.set == "test":
        pd.DataFrame(rows).to_csv(outdir / "boxes.csv", index=False)
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
