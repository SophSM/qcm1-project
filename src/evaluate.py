"""Validation evaluation at original crop resolution (TRAINING_PLAN Step 6).

    python src/evaluate.py --pred val_final --tag final

Needs predicted masks from: predict.py --set apo_val / fasc_val --out <pred>.
Reports Dice/IoU, aponeurosis thickness error (px and, where a ruler is found on the image, mm), dominant fascicle
angle error (deg), and a proxy UMUD score = mean(MT error / 1 mm, angle error / 4 deg) (FL has no proxy).
Outputs: outputs/eval_<tag>.json, outputs/eval_<tag>_{apo,fasc}.csv, outputs/figures/val_examples_{apo,fasc}_<tag>.png
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from data import load_img, load_mask  # noqa: E402
from geometry import aponeuroses, dominant_angle, fascicle_segments, thickness  # noqa: E402
from scale import detect_scale  # noqa: E402


def dice_iou(p, t):
    inter, s, union = (p & t).sum(), p.sum() + t.sum(), (p | t).sum()
    return (1.0, 1.0) if s == 0 else (2 * inter / s, inter / max(union, 1))


def eval_apo(pdir):
    sp = pd.read_csv(config.OUTPUTS / "split_apo.csv")
    boxes = pd.read_csv(pdir / "boxes_apo_val.csv").set_index("file")
    rows = []
    for f in sp[sp.fold == "val"].file:
        g = load_img(config.DIRS["apo"]["imgs"] / f)
        t = load_mask(config.DIRS["apo"]["masks"] / f, size=(g.shape[1], g.shape[0]))
        pp = np.array(Image.open(pdir / "apo" / (os.path.splitext(f)[0] + ".png"))).astype(np.float32) / 255
        p = pp > 0.5
        b = boxes.loc[f]
        sl = (slice(b.y0, b.y1), slice(b.x0, b.x1))
        d, j = dice_iou(p[sl], t[sl])
        at, ap = aponeuroses(t[sl]), aponeuroses(p[sl], prob=pp[sl])
        r = dict(file=f, dice=d, iou=j, true_ok=at is not None, pred_ok=ap is not None)
        for w in ("edge", "mid"):
            r[f"mt_true_{w}"] = thickness(at, w)[0]
            r[f"mt_pred_{w}"] = thickness(ap, w)[0]
        rgb = np.array(Image.open(config.DIRS["apo"]["imgs"] / f).convert("RGB"))
        s = detect_scale(rgb, (b.x0, b.y0, b.x1, b.y1))
        r["sy"] = s["sy"] if s["confidence"] in ("high", "medium") else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    for w in ("edge", "mid"):
        df[f"mt_err_px_{w}"] = (df[f"mt_pred_{w}"] - df[f"mt_true_{w}"]).abs()
    df["mt_err_mm"] = df.mt_err_px_edge / df.sy
    return df


def eval_fasc(pdir):
    sp = pd.read_csv(config.OUTPUTS / "split_fasc.csv")
    boxes = pd.read_csv(pdir / "boxes_fasc_val.csv").set_index("file")
    rows = []
    for f in sp[sp.fold == "val"].file:
        g = load_img(config.DIRS["fasc"]["imgs"] / f)
        t = load_mask(config.DIRS["fasc"]["masks"] / f, size=(g.shape[1], g.shape[0]))
        tt = cv2.dilate(t.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        p = np.array(Image.open(pdir / "fasc" / (os.path.splitext(f)[0] + ".png"))) > 127
        b = boxes.loc[f]
        sl = (slice(b.y0, b.y1), slice(b.x0, b.x1))
        d, j = dice_iou(p[sl], tt[sl])
        ml = max(15, int(0.04 * (b.y1 - b.y0)))
        at = dominant_angle(fascicle_segments(tt[sl], min_len=ml))
        ap = dominant_angle(fascicle_segments(p[sl], min_len=ml))
        rows.append(dict(file=f, dice=d, iou=j, ang_true=at, ang_pred=ap, ang_err=abs(at - ap)))
    return pd.DataFrame(rows)


def examples(task, df, pdir, tag, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    worst = df.nsmallest(4, "dice").file.tolist()
    rnd = df[~df.file.isin(worst)].sample(n - 4, random_state=config.SEED).file.tolist()
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.2 * n))
    boxes = pd.read_csv(pdir / f"boxes_{task}_val.csv").set_index("file")
    for row, f in zip(axes, rnd + worst):
        g = load_img(config.DIRS[task]["imgs"] / f)
        t = load_mask(config.DIRS[task]["masks"] / f, size=(g.shape[1], g.shape[0]))
        p = np.array(Image.open(pdir / task / (os.path.splitext(f)[0] + ".png"))) > 127
        b = boxes.loc[f]
        sl = (slice(b.y0, b.y1), slice(b.x0, b.x1))
        g, t, p = g[sl], t[sl], p[sl]
        ov = np.dstack([g, g, g]).copy()
        ov[t] = (0.5 * ov[t] + [0, 127, 0]).astype(np.uint8)
        ov[p] = (0.5 * ov[p] + [127, 0, 0]).astype(np.uint8)
        dsc = df.set_index("file").loc[f, "dice"]
        for a, im, title in zip(row, [g, t, p, ov], ["image", "true", "predicted", "overlay (green true, red pred)"]):
            a.imshow(im, cmap="gray")
            a.set_title(f"{f} {title}" + (f"  Dice {dsc:.2f}" if title == "predicted" else ""), fontsize=8)
            a.axis("off")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS / "figures" / f"val_examples_{task}_{tag}.png", dpi=70)
    plt.close()


def summ(x):
    x = pd.Series(x).dropna()
    return dict(mean=float(x.mean()), sd=float(x.std()), median=float(x.median()), n=int(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    pdir = config.OUTPUTS / "pred_masks" / args.pred
    res = {}
    if (pdir / "apo").exists():
        a = eval_apo(pdir)
        a.to_csv(config.OUTPUTS / f"eval_{args.tag}_apo.csv", index=False)
        both = a.true_ok & a.pred_ok
        res["apo"] = dict(dice=summ(a.dice), iou=summ(a.iou), both_bands_found=float(both.mean()),
                          true_two_bands=float(a.true_ok.mean()),
                          mt_err_px_edge=summ(a.mt_err_px_edge[both]), mt_err_px_mid=summ(a.mt_err_px_mid[both]),
                          mt_corr_edge=float(a[both][["mt_true_edge", "mt_pred_edge"]].corr().iloc[0, 1]),
                          mt_err_mm=summ(a.mt_err_mm[both]),
                          worst10=a.nsmallest(10, "dice")[["file", "dice"]].round(3).values.tolist())
        examples("apo", a, pdir, args.tag)
    if (pdir / "fasc").exists():
        fz = fasc = eval_fasc(pdir)
        fasc.to_csv(config.OUTPUTS / f"eval_{args.tag}_fasc.csv", index=False)
        res["fasc"] = dict(dice=summ(fz.dice), iou=summ(fz.iou), angle_err=summ(fz.ang_err),
                           angle_missing=int((fz.ang_true.notna() & fz.ang_pred.isna()).sum()),
                           worst10=fz.nsmallest(10, "dice")[["file", "dice"]].round(3).values.tolist())
        examples("fasc", fz, pdir, args.tag)
    if "apo" in res and "fasc" in res:
        mt, pa = res["apo"]["mt_err_mm"]["mean"], res["fasc"]["angle_err"]["mean"]
        res["proxy_umud"] = dict(mt_term=mt / 1.0, pa_term=pa / 4.0, score=(mt / 1.0 + pa / 4.0) / 2,
                                 note="MT error in mm on val images with a detectable ruler; PA proxy = dominant "
                                      "fascicle angle error; FL has no proxy (no ground truth)")
    out = config.OUTPUTS / f"eval_{args.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
