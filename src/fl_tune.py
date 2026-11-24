"""Run 2 (TRAINING_PLAN_2.md): tune the fascicle-length (FL) / pennation-angle (PA) rule on a validation proxy.

    python src/fl_tune.py                      # all images, all rules (CPU job: slurm/fl_tune.slurm)
    python src/fl_tune.py --n 10 --workers 1   # quick test on 10 images

Proxy, on the fascicle validation images (v2 models only, never v3: v3 was trained on these images):
  reference  = TRUE fascicle mask (dilated as in training) + PREDICTED aponeuroses -> run-1 rule -> FL_ref, PA_ref
  candidate  = PREDICTED fascicle probabilities + the SAME predicted aponeuroses   -> candidate rule -> FL_hat, PA_hat
  FL_err_mm  = |FL_hat - FL_ref| in mm (images with a high-confidence ruler, values clipped to 30-200 mm as in the
               submission), FL_err_rel = same in px / crop height (all images), PA_err = |PA_hat - PA_ref| (deg)
  score      = mean(FL_err_mm) / 10 / 2 + mean(PA_err) / 4 / 2      (stand-in tolerances 10 mm, 4 deg)
A candidate that finds no fascicle gets the median of that rule over the images (as the global test fallback).
Images where the predicted aponeuroses fail, or the reference has no fascicle, are skipped (counted).
The proxy only sees errors of fascicle prediction / selection, not aponeurosis or scale errors.

Inputs: outputs/pred_masks/val_v2/fasc (v2 fascicle model), outputs/pred_masks/val2_fasc_apo/apo (v2 apo A+B).
Outputs: outputs/fl_tuning.csv (one row per rule), outputs/fl_tune_raw.npz (per-image values),
         outputs/figures/fl_tuning.png, outputs/figures/fl_apo_check.png
"""
import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from data import load_mask  # noqa: E402
from geometry import aponeuroses  # noqa: E402
from measure import RANGES, RUN1, THR, aggregate, load_prob, select_fascicles  # noqa: E402
from scale import detect_scale  # noqa: E402

FAMILIES = dict(  # run-1 value first
    top_k=[7, 3, 5, 10],
    dir_tol=[8.0, 5.0, 12.0],
    min_len_frac=[0.04, 0.08, 0.12],
    agg=["median", "wmedian", "angle"],
    min_prob=[None, 0.5, 0.6],
    mid_frac=[None, 0.8],
    shrink_w=[None, 0.5, 0.75],
)
FAMILY_LETTER = dict(top_k="A", dir_tol="B", min_len_frac="C", agg="D", min_prob="E", mid_frac="F", shrink_w="G")
SELECT_KEYS = ["dir_tol", "min_len_frac", "min_prob", "mid_frac"]  # change which segments are found
PRED_FASC, PRED_APO = "val_v2", "val2_fasc_apo"


def all_rules():
    keys = list(FAMILIES)
    return [dict(zip(keys, v), pa_from=None) for v in itertools.product(*FAMILIES.values())]


def rule_name(r):
    f = lambda v: "none" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))  # noqa: E731
    return (f"k{r['top_k']}_dir{f(r['dir_tol'])}_len{f(r['min_len_frac'])}_{r['agg']}_p{f(r['min_prob'])}"
            f"_mid{f(r['mid_frac'])}_w{f(r['shrink_w'])}" + ("_paRun1" if r.get("pa_from") == "run1" else ""))


def n_changes(r):
    return sum(r[k] != RUN1[k] for k in FAMILIES)


def px_copy(fas):
    """Same fascicles with per-fascicle FL in pixels (from the stored intersection points)."""
    return [dict(f, fl=float(np.hypot(f["hit_d"][0] - f["hit_s"][0], f["hit_d"][1] - f["hit_s"][1]))) for f in fas]


def clip(v, key):
    return v if not np.isfinite(v) else float(np.clip(v, *RANGES[key]))


def work(f):
    """All rules on one validation image -> dict with per-rule arrays (pa, fl_mm, fl_px) + reference."""
    rules = all_rules()
    boxes = pd.read_csv(config.OUTPUTS / "pred_masks" / PRED_APO / "boxes_fasc_val.csv").set_index("file")
    b = boxes.loc[f]
    box = (int(b.x0), int(b.y0), int(b.x1), int(b.y1))
    ch = box[3] - box[1]
    out = dict(file=f, ch=ch, apo_ok=False, ref_ok=False, sx=np.nan, sy=np.nan, conf="")
    rgb = np.array(Image.open(config.DIRS["fasc"]["imgs"] / f).convert("RGB"))
    s = detect_scale(rgb, box)
    out.update(sx=s["sx"], sy=s["sy"], conf=s["confidence"])
    has_mm = s["confidence"] == "high"
    sx, sy = (s["sx"], s["sy"]) if has_mm else (1.0, 1.0)
    apo_p = load_prob(config.OUTPUTS / "pred_masks" / PRED_APO / "apo", f)
    fasc_p = load_prob(config.OUTPUTS / "pred_masks" / PRED_FASC / "fasc", f)
    apo = aponeuroses(apo_p > THR, prob=apo_p, pick="fasc", fasc=fasc_p > THR)  # as measure.measure_image
    if apo is None:
        return out
    out["apo_ok"] = True
    H, W = apo_p.shape
    t = load_mask(config.DIRS["fasc"]["masks"] / f, size=(W, H))
    tt = cv2.dilate(t.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
    keep = np.zeros_like(tt)
    keep[box[1]:box[3], box[0]:box[2]] = True
    tt = (tt & keep).astype(np.float32)
    fr = select_fascicles(tt, apo, box, sx, sy, RUN1)
    pa_r, fl_r = aggregate(fr, apo, sx, sy, RUN1)
    out.update(ref_ok=bool(fr), pa_ref=pa_r, fl_ref_px=aggregate(px_copy(fr), apo, 1, 1, RUN1)[1],
               fl_ref_mm=clip(fl_r, "fl_mm") if has_mm else np.nan, n_ref=len(fr))
    cache = {}
    vals = np.full((len(rules), 3), np.nan)
    for i, r in enumerate(rules):
        key = tuple(r[k] for k in SELECT_KEYS)
        if key not in cache:
            sel = dict(RUN1, **{k: r[k] for k in SELECT_KEYS})
            cache[key] = select_fascicles(fasc_p, apo, box, sx, sy, sel, max_k=max(FAMILIES["top_k"]))
        fas = cache[key][: r["top_k"]]  # sorted by length, so the k longest = first k
        pa, fl = aggregate(fas, apo, sx, sy, r)
        vals[i] = (pa, clip(fl, "fl_mm") if has_mm else np.nan, aggregate(px_copy(fas), apo, 1, 1, r)[1])
    out["vals"] = vals
    return out


def score_table(res, rules):
    ok = [r for r in res if r["apo_ok"] and r["ref_ok"]]
    V = np.stack([r["vals"] for r in ok])                    # images x rules x (pa, fl_mm, fl_px)
    pa_ref = np.array([r["pa_ref"] for r in ok])
    flmm_ref = np.array([r["fl_ref_mm"] for r in ok])
    flpx_ref = np.array([r["fl_ref_px"] for r in ok])
    ch = np.array([r["ch"] for r in ok], float)
    mm = np.isfinite(flmm_ref)
    i_run1 = next(i for i, r in enumerate(rules) if all(r[k] == RUN1[k] for k in FAMILIES))
    rows, per_img = [], {}
    variants = [(i, r) for i, r in enumerate(rules)]
    for i, r in variants:
        for pa_from in (None, "run1"):
            if pa_from == "run1" and all(r[k] == RUN1[k] for k in ("top_k",) + tuple(SELECT_KEYS)):
                continue  # PA identical to run 1 anyway
            pa = V[:, i_run1 if pa_from else i, 0].copy()
            flmm, flpx = V[:, i, 1].copy(), V[:, i, 2].copy()
            miss = np.isnan(V[:, i, 2])
            for a in (pa, flmm, flpx):  # global fallback: median of this rule over the images
                a[np.isnan(a)] = np.nanmedian(a)
            pa_err = np.abs(pa - pa_ref)
            fl_err_mm = np.abs(flmm - flmm_ref)[mm]
            fl_err_rel = np.abs(flpx - flpx_ref) / ch
            rr = dict(r, pa_from=pa_from)
            name = rule_name(rr)
            per_img[name] = (pa_err, np.where(mm, np.abs(flmm - flmm_ref), np.nan))
            rows.append(dict(rule=name, n_changes=n_changes(r) + (pa_from is not None), **{k: r[k] for k in FAMILIES},
                             pa_from=pa_from,
                             score=0.5 * fl_err_mm.mean() / 10 + 0.5 * pa_err.mean() / 4,
                             fl_term=fl_err_mm.mean() / 10, pa_term=pa_err.mean() / 4,
                             fl_err_mm_mean=fl_err_mm.mean(), fl_err_mm_median=np.median(fl_err_mm),
                             fl_err_rel_mean=fl_err_rel.mean(), fl_err_rel_median=np.median(fl_err_rel),
                             pa_err_mean=pa_err.mean(), pa_err_median=np.median(pa_err),
                             n_missing=int(miss.sum())))
    df = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    n = dict(n_images=len(res), apo_failed=sum(not r["apo_ok"] for r in res),
             ref_missing=sum(r["apo_ok"] and not r["ref_ok"] for r in res), n_scored=len(ok), n_mm=int(mm.sum()))
    return df, per_img, n, [r["file"] for r in ok]


def family_figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = df[(df.n_changes == 0)].iloc[0]
    fams = list(FAMILIES)
    fig, axes = plt.subplots(1, len(fams), figsize=(2.4 * len(fams), 3.4), sharey=True)
    for a, k in zip(axes, fams):
        sub = []
        for v in FAMILIES[k]:
            m = (df.n_changes == (0 if v == RUN1[k] else 1)) & df.pa_from.isna()
            m &= df[k].apply(lambda x: x == v or (x is None and v is None) or (pd.isna(x) and v is None))
            sub.append((("none" if v is None else str(v)), df[m].score.iloc[0]))
        labels, sc = zip(*sub)
        a.bar(range(len(sc)), sc, color=["#8c8c8c"] + ["#4472c4"] * (len(sc) - 1), width=0.65)
        a.set_xticks(range(len(sc)), labels, fontsize=8)
        a.set_title(f"{FAMILY_LETTER[k]}. {k}", fontsize=9)
        a.axhline(base.score, color="#8c8c8c", lw=0.8, ls="--")
        a.axhline(base.score * 0.97, color="#c00000", lw=0.8, ls=":")
        a.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("proxy score (lower = better)")
    lo = df.score.min()
    axes[0].set_ylim(lo * 0.9, max(df[df.n_changes <= 1].score.max(), base.score) * 1.03)
    fig.suptitle("FL/PA rule tuning, one family at a time (grey = run 1; dashed = run 1, dotted = 3 % better)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(config.OUTPUTS / "figures" / "fl_tuning.png", dpi=100)
    plt.close()


def apo_check_figure(files):
    """3 fascicle-validation images with the predicted (v2 A+B) aponeurosis bands overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(files), figsize=(6 * len(files), 4.5))
    for a, f in zip(np.atleast_1d(axes), files):
        g = np.array(Image.open(config.DIRS["fasc"]["imgs"] / f).convert("L"))
        p = load_prob(config.OUTPUTS / "pred_masks" / PRED_APO / "apo", f)
        fp = load_prob(config.OUTPUTS / "pred_masks" / PRED_FASC / "fasc", f)
        ov = np.dstack([g, g, g]).astype(np.float32)
        ov[p > THR] = 0.5 * ov[p > THR] + [127, 0, 0]
        ov[fp > THR] = 0.5 * ov[fp > THR] + [0, 127, 0]
        a.imshow(ov.astype(np.uint8))
        apo = aponeuroses(p > THR, prob=p, pick="fasc", fasc=fp > THR)
        if apo is not None:
            xs = np.linspace(0, g.shape[1], 200)
            a.plot(xs, apo["sup_low_lin"](xs), "c-", lw=1)
            a.plot(xs, apo["deep_up_lin"](xs), "y-", lw=1)
        a.set_xlim(0, g.shape[1])
        a.set_ylim(g.shape[0], 0)
        a.set_title(f"{f}: red = predicted apo, green = predicted fasc, lines = measured pair", fontsize=8)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS / "figures" / "fl_apo_check.png", dpi=80)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="only the first n images (quick test)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="fl_tuning.csv")
    args = ap.parse_args()
    sp = pd.read_csv(config.OUTPUTS / "split_fasc.csv")
    files = sp[sp.fold == "val"].file.tolist()
    if args.n:
        files = files[:: max(1, len(files) // args.n)][: args.n]
    rules = all_rules()
    print(len(files), "images,", len(rules), "rules", flush=True)
    if args.workers > 1:
        with Pool(args.workers) as pool:
            res = pool.map(work, files, chunksize=2)
    else:
        res = [work(f) for f in files]
    df, per_img, n, ok_files = score_table(res, rules)
    print(n, flush=True)
    df.to_csv(config.OUTPUTS / args.out, index=False)
    if not args.n:
        unit = sp.set_index("file").loc[ok_files, "unit"].values
        np.savez_compressed(config.OUTPUTS / "fl_tune_raw.npz", files=np.array(ok_files), unit=unit,
                            names=np.array(list(per_img)),
                            pa_err=np.stack([v[0] for v in per_img.values()]),
                            fl_err_mm=np.stack([v[1] for v in per_img.values()]),
                            meta=np.array([str(n)]))
        family_figure(df)
        apo_check_figure(files[::140][:3])
    pd.set_option("display.width", 250)
    print(df.head(15).drop(columns=list(FAMILIES)).to_string())
    print(df[df.n_changes == 0].drop(columns=list(FAMILIES)).to_string())


if __name__ == "__main__":
    main()
