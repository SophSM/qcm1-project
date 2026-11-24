"""Train/validation split that keeps near-identical frames and duplicates together.

1. For every training image: md5 of the grey pixels (exact duplicates) and a 64x64 thumbnail (near duplicates,
   Pearson r > NEAR_DUP_R).
2. Similarity of neighbouring filenames -> if consecutive files are often near-identical, neighbouring files with
   r > CLIP_R are joined into "clips" (content-defined contiguous blocks).
3. Union-find over {block membership, exact dups, near dups} -> units; ~15 % of images (by unit) go to validation.
Filenames are used only for the split, never at prediction time.

Usage (CPU job): python src/split.py   -> outputs/split_apo.csv, outputs/split_fasc.csv, outputs/split_stats.txt
"""
import hashlib
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from data import list_images, load_img  # noqa: E402

BLOCK = 50
NEAR_DUP_R = 0.99
CLIP_R = 0.9


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def thumbs(task):
    d = config.DIRS[task]["imgs"]
    files = list_images(d)
    md5s, th, sizes = [], [], []
    for f in files:
        g = load_img(d / f)
        md5s.append(hashlib.md5(g.tobytes()).hexdigest())
        t = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        t = (t - t.mean()) / (t.std() + 1e-6)
        th.append(t)
        sizes.append(g.shape)
    return files, md5s, np.array(th), sizes


def split_task(task, log):
    files, md5s, th, sizes = thumbs(task)
    n = len(files)
    R = th @ th.T / th.shape[1]
    code = np.array([h * 10000 + w for h, w in sizes])
    same_size = code[:, None] == code[None, :]
    near = (R > NEAR_DUP_R) & same_size
    np.fill_diagonal(near, False)
    consec = np.array([R[i, i + 1] for i in range(n - 1)])
    exact_groups = pd.Series(md5s).value_counts()
    n_exact = int((exact_groups > 1).sum())
    n_exact_imgs = int(exact_groups[exact_groups > 1].sum())
    frac_consec = float((consec > 0.9).mean())
    use_blocks = frac_consec > 0.2
    log.append(f"[{task}] n={n}; exact-duplicate groups={n_exact} ({n_exact_imgs} images); "
               f"near-duplicate pairs (r>{NEAR_DUP_R})={int(near.sum() // 2)}; "
               f"consecutive files r>0.9: {frac_consec:.1%} (median r={np.median(consec):.3f}); "
               f"split mode={'content-defined clips (neighbour r>%.2f)' % CLIP_R if use_blocks else 'random'}")
    uf = UF(n)
    if use_blocks:
        # "clips": join neighbouring files that look like consecutive video frames (content-defined blocks).
        # Fixed blocks of BLOCK files were tried first, but chained through the duplicated copies into one
        # giant unit (see DECISIONS.md).
        for i in range(1, n):
            if consec[i - 1] > CLIP_R and sizes[i] == sizes[i - 1]:
                uf.union(i, i - 1)
    by_md5 = {}
    for i, h in enumerate(md5s):
        if h in by_md5:
            uf.union(i, by_md5[h])
        by_md5[h] = i
    ii, jj = np.where(np.triu(near))
    for a, b in zip(ii, jj):
        uf.union(a, b)
    units = np.array([uf.find(i) for i in range(n)])
    rng = np.random.default_rng(config.SEED)
    uniq = rng.permutation(np.unique(units))
    sizes_u = pd.Series(units).value_counts()
    val_units, cnt = set(), 0
    for u in uniq:
        if cnt >= config.VAL_FRACTION * n:
            break
        if sizes_u[u] > 0.08 * n:  # never put a giant merged unit into validation
            continue
        val_units.add(u)
        cnt += sizes_u[u]
    fold = np.where(np.isin(units, list(val_units)), "val", "train")
    log.append(f"[{task}] units={len(uniq)}, largest unit={sizes_u.max()} images; val={int((fold == 'val').sum())} "
               f"({(fold == 'val').mean():.1%}), train={int((fold == 'train').sum())}")
    df = pd.DataFrame(dict(file=files, md5=md5s, unit=units, fold=fold))
    df.to_csv(config.OUTPUTS / f"split_{task}.csv", index=False)
    return files, consec


def neighbour_figure(task, files, consec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = config.DIRS[task]["imgs"]
    starts = [100, 1000] if len(files) > 1100 else [100, 600]
    fig, axes = plt.subplots(len(starts), 4, figsize=(16, 3 * len(starts)))
    for row, s in zip(axes, starts):
        for k, a in enumerate(row):
            a.imshow(load_img(d / files[s + k]), cmap="gray")
            a.set_title(f"{files[s + k]}" + (f"  r(next)={consec[s + k]:.2f}" if s + k < len(consec) else ""),
                        fontsize=8)
            a.axis("off")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS / "figures" / f"neighbour_frames_{task}.png", dpi=80)


def main():
    os.makedirs(config.OUTPUTS / "figures", exist_ok=True)
    log = []
    for task in ("apo", "fasc"):
        files, consec = split_task(task, log)
        neighbour_figure(task, files, consec)
    txt = "\n".join(log)
    print(txt)
    (config.OUTPUTS / "split_stats.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
