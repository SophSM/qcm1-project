"""Model input construction (identical for training and test) + training cache.

Input = 3 channels built from the cropped, resized grey image:
  ch1 grey, ch2 CLAHE, ch3 ridge filter (Sato, bright ridges).
Variants (experiment, TRAINING_PLAN §5):
  A  grey x3                      (built at load time from the B cache)
  B  grey + CLAHE + ridge
  C  B with mild grain removal (bilateral) before CLAHE/ridge
  D  B with stronger grain removal (non-local means) before CLAHE/ridge

Geometry of the cache:
  apo : crop resized to 512x512
  fasc: crop resized to height 768 (aspect kept)

Usage (CPU job): python src/preprocess.py --task fasc --variants B C D --workers 16
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import sato

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from data import load_img, load_mask  # noqa: E402

FASC_H = 768
APO_SIZE = 512


def denoise(g, variant):
    if variant == "C":
        return cv2.bilateralFilter(g, d=5, sigmaColor=20, sigmaSpace=5)
    if variant == "D":
        return cv2.fastNlMeansDenoising(g, None, h=12, templateWindowSize=7, searchWindowSize=21)
    return g


def make_input(g, variant="B"):
    """g: uint8 grey image (already cropped + resized). Returns HxWx3 uint8."""
    if variant == "A":
        return np.dstack([g, g, g])
    d = denoise(g, variant)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(d)
    r = sato(d.astype(np.float32) / 255.0, sigmas=(1, 2, 3), black_ridges=False)
    hi = np.percentile(r, 99.5)
    r = np.clip(r / (hi + 1e-6), 0, 1)
    return np.dstack([g, clahe, (r * 255).astype(np.uint8)])


def target_size(task, w, h):
    if task == "apo":
        return APO_SIZE, APO_SIZE
    s = FASC_H / h
    return max(32, int(round(w * s))), FASC_H


def crop_resize_gray(g, box, task):
    x0, y0, x1, y1 = box
    c = g[y0:y1, x0:x1]
    tw, th = target_size(task, c.shape[1], c.shape[0])
    return cv2.resize(c, (tw, th), interpolation=cv2.INTER_AREA)


def prep_mask(m, box, task):
    """m: bool mask at original image size."""
    if task == "fasc":  # thicken thin lines to ~5 px at original resolution
        m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    x0, y0, x1, y1 = box
    c = m[y0:y1, x0:x1].astype(np.uint8) * 255
    tw, th = target_size(task, c.shape[1], c.shape[0])
    return cv2.resize(c, (tw, th), interpolation=cv2.INTER_NEAREST)


def cache_dir(task, variant):
    return config.OUTPUTS / "cache" / task / variant


def _work(args):
    task, f, box, variants = args
    img_p = config.DIRS[task]["imgs"] / f
    msk_p = config.DIRS[task]["masks"] / f
    g = load_img(img_p)
    m = load_mask(msk_p, size=(g.shape[1], g.shape[0]))
    gc = crop_resize_gray(g, box, task)
    stem = os.path.splitext(f)[0] + ".png"
    mdir = config.OUTPUTS / "cache" / task / "masks"
    if not (mdir / stem).exists():
        Image.fromarray(prep_mask(m, box, task)).save(mdir / stem)
    for v in variants:
        p = cache_dir(task, v) / stem
        if not p.exists():
            Image.fromarray(make_input(gc, v)).save(p)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--variants", nargs="+", default=["B"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    crops = pd.read_csv(config.OUTPUTS / "crops.csv")
    crops = crops[crops.set == args.task]
    os.makedirs(config.OUTPUTS / "cache" / args.task / "masks", exist_ok=True)
    for v in args.variants:
        os.makedirs(cache_dir(args.task, v), exist_ok=True)
    jobs = [(args.task, r.file, (r.x0, r.y0, r.x1, r.y1), args.variants) for r in crops.itertuples()]
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            if i % 200 == 0:
                print(args.task, i, len(jobs), flush=True)
    print("done", args.task, args.variants)


def example_figure():
    """Preprocessing steps on 3 fascicle images -> outputs/figures/preprocess_examples.png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    crops = pd.read_csv(config.OUTPUTS / "crops.csv")
    crops = crops[crops.set == "fasc"].groupby("W").head(1).head(3)
    fig, axes = plt.subplots(len(crops), 5, figsize=(20, 3.6 * len(crops)))
    for row, r in zip(axes, crops.itertuples()):
        g = crop_resize_gray(load_img(config.DIRS["fasc"]["imgs"] / r.file), (r.x0, r.y0, r.x1, r.y1), "fasc")
        b, c, d = make_input(g, "B"), make_input(g, "C"), make_input(g, "D")
        for a, im, t in zip(row, [g, b[..., 1], b[..., 2], c[..., 2], d[..., 2]],
                            ["grey (cropped)", "CLAHE", "ridge (B)", "ridge after mild denoise (C)",
                             "ridge after strong denoise (D)"]):
            a.imshow(im, cmap="gray")
            a.set_title(f"{r.file}: {t}", fontsize=8)
            a.axis("off")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS / "figures" / "preprocess_examples.png", dpi=90)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--figure":
        example_figure()
    else:
        main()
