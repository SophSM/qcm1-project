"""Find the ultrasound scan area of an image (content-based, no filenames).

Idea: the scan is a large rectangle of grey (unsaturated), textured (speckle) pixels. Text, rulers, colour bars and
black borders are either saturated, flat or sparse. We compute a "textured" map, then take the longest run of
columns (and then rows) in which a good fraction of pixels is textured.

Usage (CPU job):  python src/crop.py            -> outputs/crops.csv + outputs/figures/crop_check.png
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from data import list_images  # noqa: E402


def _longest_run(mask, min_gap=6):
    """Start/end (inclusive) of the longest run of True, bridging gaps shorter than min_gap."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return 0, len(mask) - 1
    runs, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            runs.append((s, prev))
            s = i
        prev = i
    runs.append((s, prev))
    return max(runs, key=lambda r: r[1] - r[0])


def _extend(rowfrac, y, step, weak=0.12, dead=0.03, max_dead=10):
    """Move from row y in direction step while dead (texture-free) stretches stay shorter than max_dead rows."""
    best, n_dead, i = y, 0, y + step
    while 0 <= i < len(rowfrac):
        if rowfrac[i] < dead:
            n_dead += 1
            if n_dead > max_dead:
                break
        else:
            n_dead = 0
            if rowfrac[i] > weak:
                best = i
        i += step
    return best


def textured_map(rgb):
    rgb = rgb.astype(np.float32)
    g = rgb.mean(2)
    sat = rgb.max(2) - rgb.min(2)
    m = cv2.blur(g, (9, 9))
    sd = np.sqrt(np.clip(cv2.blur(g * g, (9, 9)) - m * m, 0, None))
    return (sd > 2.0) & (sat < 40) & (m > 3)


def find_crop(rgb, frac=0.35, margin=2):
    """Return (x0, y0, x1, y1) of the scan area; x1/y1 exclusive."""
    H, W = rgb.shape[:2]
    t = textured_map(rgb)
    # columns: long vertical structure needed; rows: within those columns
    colfrac = t.mean(0)
    x0, x1 = _longest_run(colfrac > frac)
    # rows: strict core, then extend through dark (little speckle) muscle rows, but stop at dead bands
    rowfrac = t[:, x0:x1 + 1].mean(1)
    y0, y1 = _longest_run(rowfrac > frac)
    y0, y1 = _extend(rowfrac, y0, -1), _extend(rowfrac, y1, +1)
    # refine columns inside the row range (removes rulers above/below)
    colfrac = t[y0:y1 + 1].mean(0)
    x0, x1 = _longest_run(colfrac > frac)
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(W, x1 + 1 + margin), min(H, y1 + 1 + margin)
    # degenerate result -> whole image
    if (x1 - x0) < 0.25 * W or (y1 - y0) < 0.25 * H:
        return 0, 0, W, H
    return int(x0), int(y0), int(x1), int(y1)


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def all_images():
    out = []
    for task in ("apo", "fasc"):
        d = config.DIRS[task]["imgs"]
        out += [(task, f, d / f) for f in list_images(d)]
    out += [("test", f, config.TEST_DIR / f) for f in list_images(config.TEST_DIR)]
    return out


def _crop_row(item):
    set_, f, p = item
    rgb = load_rgb(p)
    x0, y0, x1, y1 = find_crop(rgb)
    return dict(set=set_, file=f, W=rgb.shape[1], H=rgb.shape[0], x0=x0, y0=y0, x1=x1, y1=y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    items = all_images()
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)]
    from multiprocessing import Pool
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
        rows = pool.map(_crop_row, items, chunksize=8)
    df = pd.DataFrame(rows)
    out = config.OUTPUTS / ("crops.csv" if not args.limit else "crops_sample.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df))
    check_figure(df, fname="crop_check.png" if not args.limit else "crop_check_sample.png")


def check_figure(df, n=40, fname="crop_check.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(config.SEED)
    # one per (set, size) first, then random fill
    pick = df.groupby(["set", "W", "H"]).head(1).index.tolist()
    rest = [i for i in df.index if i not in pick]
    pick += list(rng.choice(rest, min(len(rest), max(0, n - len(pick))), replace=False)) if rest else []
    pick = pick[:n]
    cols = 8
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.4))
    for a, i in zip(axes.flat, pick):
        r = df.loc[i]
        d = config.TEST_DIR if r.set == "test" else config.DIRS[r.set]["imgs"]
        a.imshow(load_rgb(d / r.file))
        a.add_patch(plt.Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, ec="lime", lw=1.2))
        a.set_title(f"{r.set} {r.file} {r.W}x{r.H}", fontsize=6)
        a.axis("off")
    for a in axes.flat[len(pick):]:
        a.axis("off")
    plt.tight_layout()
    os.makedirs(config.OUTPUTS / "figures", exist_ok=True)
    plt.savefig(config.OUTPUTS / "figures" / fname, dpi=110)


if __name__ == "__main__":
    main()
