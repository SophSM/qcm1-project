"""Predicted masks -> pa_deg, fl_mm, mt_mm per test image -> submission.

    python src/measure.py --pred test --out submission_v1_baseline.csv
    python src/measure.py --pred test_v3 --rule <name> --out submission_v4_<name>_v3models.csv --same_mt submission_v3_alldata.csv

Method (copies the expert protocol, COMPETITION_INFO.md):
  mt : perpendicular distance between the lower edge of the superficial and the upper edge of the deep aponeurosis
       at 25/50/75 % of the width where both are detected, averaged.
  pa : angle between each fascicle line and the deep aponeurosis (straight-line fit of its upper edge), measured
       in the image plane; median over the 7 longest fascicles between the bands.
  fl : distance between the two aponeurosis edges (straight-line fits, extended beyond the image) along each
       extended fascicle line, median over the same fascicles.
Pixel -> mm: outputs/scales.csv (src/scale.py).
failed detections are replaced by the median of the confidently measured images, values outside the official
test ranges are clipped to the range (global rules).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from data import list_images  # noqa: E402
from geometry import (aponeuroses, dominant_angle, fascicle_segments, line_curve_intersection,  # noqa: E402
                      pennation_and_length, perp_distance, thickness, weighted_median)

RANGES = dict(pa_deg=(5, 45), fl_mm=(30, 200), mt_mm=(10, 50))
TOP_K = 7
DIR_TOL = 8.0  # degrees around the dominant fascicle direction
THR = 0.5

# Fascicle rules (run 2, TRAINING_PLAN_2.md). "run1" reproduces submissions v1-v3 exactly.
#   top_k        number of longest fascicles measured
#   dir_tol      keep fascicles within this many degrees of the dominant (length-weighted median) direction
#   min_len_frac minimum segment length as a fraction of the crop height (at least 15 px)
#   agg          FL aggregation: "median" of per-fascicle FL | "wmedian" (length-weighted) |
#                "angle" (one line with the median fascicle direction through the band centre)
#   min_prob     drop segments whose mean probability along the skeleton is below this (None = off)
#   mid_frac     keep only segments whose centre lies in the middle `mid_frac` of the crop width (None = all)
#   shrink_w     FL = w * FL + (1 - w) * MT_centre / sin(PA) (None = off)
#   pa_from      None = PA from this rule's fascicles; "run1" = PA from the run-1 rule (only FL changes)
RUN1 = dict(top_k=TOP_K, dir_tol=DIR_TOL, min_len_frac=0.04, agg="median", min_prob=None, mid_frac=None,
            shrink_w=None, pa_from=None)
RULES = {"run1": RUN1}
# Run 2: best proxy rule with <= 3 changes (outputs/fl_tuning.csv): 10 fascicles, +-12 deg direction filter, FL only
# (PA from run 1). Proxy gain 2.3 % (< 3 % = noise; clip bootstrap 95 % CI includes 0) -> exploratory candidate.
RULES["k10dir12"] = dict(RUN1, top_k=10, dir_tol=12.0, pa_from="run1")


def load_prob(d, name):
    return np.array(Image.open(d / (os.path.splitext(name)[0] + ".png"))).astype(np.float32) / 255.0


def select_fascicles(fasc_p, apo, box, sx, sy, rule, max_k=None):
    """Fascicle lines (dicts from pennation_and_length) for one image under `rule`: threshold 0.5, retry 0.3."""
    x0, y0, x1, y1 = box
    ch, cw = y1 - y0, x1 - x0
    fas = []
    for thr in (THR, 0.3):  # weak fascicles (e.g. new scanner): retry at a lower threshold
        segs = fascicle_segments(fasc_p > thr, min_len=max(15, int(rule["min_len_frac"] * ch)),
                                 prob=fasc_p if rule["min_prob"] is not None else None)
        if rule["min_prob"] is not None:
            segs = [g for g in segs if g["p"] >= rule["min_prob"]]
        if rule["mid_frac"] is not None:
            m = 0.5 * (1 - rule["mid_frac"]) * cw
            segs = [g for g in segs if x0 + m <= g["c"][0] <= x1 - m]
        dom = dominant_angle(segs)  # keep the dominant fascicle direction only (drops crossing artefacts)
        segs = [g for g in segs if abs(g["angle"] - dom) <= rule["dir_tol"]]
        fas = pennation_and_length(segs, apo, sx=sx, sy=sy, max_segs=max_k or rule["top_k"])
        if fas:
            break
    return fas


def centre_x(apo):
    a, b = apo["x_overlap"]
    if b - a < 10:  # as geometry.thickness
        a = min(apo["sup"]["cols"].min(), apo["deep"]["cols"].min())
        b = max(apo["sup"]["cols"].max(), apo["deep"]["cols"].max())
    return 0.5 * (a + b)


def fl_along_angle(apo, ang, sx, sy):
    """Length between the aponeurosis lines along one line of image-plane angle `ang` through the band centre."""
    sup, deep = apo["sup_low_lin"], apo["deep_up_lin"]
    xc = centre_x(apo)
    c = np.array([xc, 0.5 * (sup(xc) + deep(xc))])
    d = np.array([np.cos(np.radians(ang)), -np.sin(np.radians(ang))])
    hs, hd = line_curve_intersection(c, d, sup), line_curve_intersection(c, d, deep)
    if hs is None or hd is None:
        return np.nan
    return float(np.hypot((hd[1] - hs[1]) / sx, (hd[2] - hs[2]) / sy))


def aggregate(fas, apo, sx, sy, rule):
    """(pa_deg, fl) from the selected fascicles (fl in the units given by sx, sy)."""
    if not fas:
        return np.nan, np.nan
    pa = float(np.median([f["pa"] for f in fas]))
    if rule["agg"] == "median":
        fl = float(np.median([f["fl"] for f in fas]))
    elif rule["agg"] == "wmedian":
        fl = weighted_median([f["fl"] for f in fas], [f["length"] for f in fas])
    elif rule["agg"] == "angle":
        fl = fl_along_angle(apo, float(np.median([f["seg"]["angle"] for f in fas])), sx, sy)
    else:
        raise ValueError(rule["agg"])
    if rule["shrink_w"] is not None and np.isfinite(fl) and pa > 0:
        sup, deep = apo["sup_low_lin"], apo["deep_up_lin"]
        fl_geo = perp_distance(sup, deep, centre_x(apo), sx, sy) / np.sin(np.radians(pa))
        fl = rule["shrink_w"] * fl + (1 - rule["shrink_w"]) * fl_geo
    return pa, fl


def measure_image(apo_p, fasc_p, box, sx, sy, rule=None):
    rule = rule or RUN1
    apo = aponeuroses(apo_p > THR, prob=apo_p, pick="fasc", fasc=fasc_p > THR)
    r = dict(mt_px=np.nan, mt_mm=np.nan, pa_deg=np.nan, fl_mm=np.nan, n_fasc=0, apo_ok=apo is not None)
    if apo is None:
        return r, apo, []
    r["mt_px"], _ = thickness(apo, "edge")
    r["mt_mm"], _ = thickness(apo, "edge", sx=sx, sy=sy)
    fas = select_fascicles(fasc_p, apo, box, sx, sy, rule)
    r["n_fasc"] = len(fas)
    r["pa_deg"], r["fl_mm"] = aggregate(fas, apo, sx, sy, rule)
    if rule["pa_from"] == "run1":
        r["pa_deg"] = aggregate(select_fascicles(fasc_p, apo, box, sx, sy, RUN1), apo, sx, sy, RUN1)[0]
    return r, apo, fas


def stability_groups(names, test_dir, r_thr=0.9):
    """Group near-identical test frames by image content (32x32 thumbnail correlation, same size)."""
    import cv2
    th, sizes = [], []
    for n in names:
        g = np.array(Image.open(test_dir / n).convert("L"))
        t = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        th.append((t - t.mean()) / (t.std() + 1e-6))
        sizes.append(g.shape)
    th = np.array(th)
    R = th @ th.T / th.shape[1]
    code = np.array([h * 10000 + w for h, w in sizes])
    adj = (R > r_thr) & (code[:, None] == code[None, :])
    lab = -np.ones(len(names), int)
    k = 0
    for i in range(len(names)):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], k
        while stack:
            j = stack.pop()
            for nb in np.where(adj[j] & (lab < 0))[0]:
                lab[nb] = k
                stack.append(nb)
        k += 1
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="test")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--rule", default="run1", choices=sorted(RULES), help="fascicle rule (default = run 1)")
    ap.add_argument("--same_mt", default=None, help="assert that mt_mm equals this existing submission")
    args = ap.parse_args()
    rule = RULES[args.rule]
    print("rule", args.rule, rule)
    pdir = config.OUTPUTS / "pred_masks" / args.pred
    scales = pd.read_csv(config.OUTPUTS / "scales.csv").set_index("image_id")
    boxes = pd.read_csv(pdir / "boxes.csv").set_index("file")
    sample = pd.read_csv(config.SAMPLE_SUBMISSION, sep=";", encoding="utf-8-sig")
    rows, keep = [], {}
    for name in list_images(config.TEST_DIR):
        b = boxes.loc[name]
        sc = scales.loc[name]
        r, apo, fas = measure_image(load_prob(pdir / "apo", name), load_prob(pdir / "fasc", name),
                                    (b.x0, b.y0, b.x1, b.y1), sc.sx, sc.sy, rule)
        rows.append(dict(image_id=name, W=b.W, H=b.H, scale_conf=sc.confidence, sx=sc.sx, sy=sc.sy, **r))
        keep[name] = (apo, fas)
    df = pd.DataFrame(rows)
    # flags + replacement
    for col, (lo, hi) in RANGES.items():
        bad = df[col].isna() | (df[col] < lo) | (df[col] > hi)
        df[col + "_flag"] = np.where(df[col].isna(), "missing", np.where(bad, "out_of_range", ""))
        ok = ~bad & ((df.scale_conf != "low") if col != "pa_deg" else True)
        fill = df.loc[ok, col].median()
        df[col + "_raw"] = df[col]
        miss = df[col].isna()
        df.loc[miss, col] = fill                  # failed detection -> median of confident images
        df[col] = df[col].clip(lo, hi)            # out of the official range -> nearest bound
        print(f"{col}: {int(miss.sum())} missing -> median {fill:.3f}; "
              f"{int(df[col + '_flag'].eq('out_of_range').sum())} out of range -> clipped to [{lo}, {hi}]")
    df["group"] = stability_groups(df.image_id.tolist(), config.TEST_DIR)
    df.to_csv(config.OUTPUTS / ("measurements_" + os.path.splitext(args.out)[0] + ".csv"), index=False)
    sub = df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
    for c in ("pa_deg", "fl_mm", "mt_mm"):
        sub[c] = sub[c].round(3)
    sub.to_csv(config.OUTPUTS / args.out, index=False, float_format="%.3f")
    check_submission(config.OUTPUTS / args.out, sample)
    if args.same_mt:
        ref = pd.read_csv(config.OUTPUTS / args.same_mt).set_index("image_id").mt_mm
        new = pd.read_csv(config.OUTPUTS / args.out).set_index("image_id").mt_mm
        assert (new.loc[ref.index] == ref).all(), "mt_mm differs from " + args.same_mt
        print("MT identical to", args.same_mt)
    stab = df.groupby("group").filter(lambda g: len(g) >= 2).groupby("group")[["pa_deg", "fl_mm", "mt_mm"]].std()
    print(f"stability: {len(stab)} groups of near-identical frames; mean within-group SD: "
          + ", ".join(f"{c} {stab[c].mean():.2f}" for c in stab.columns))
    if args.figs:
        figures(df, keep, pdir, os.path.splitext(args.out)[0])


def check_submission(path, sample):
    """Same header as sample_submission.csv (comma-separated), one row per test image (309), no missing values."""
    s = pd.read_csv(path)  # comma-separated (Kaggle parser; the sample file uses ";")
    assert list(s.columns) == list(sample.columns) == ["image_id", "pa_deg", "fl_mm", "mt_mm"], s.columns
    names = list_images(config.TEST_DIR)
    assert len(s) == len(names) == 309, len(s)
    assert sorted(s.image_id) == sorted(names) and s.image_id.is_unique
    assert s.notna().all().all()
    print("submission check OK:", path, len(s), "rows")


def figures(df, keep, pdir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = config.OUTPUTS / "figures"
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    for a, c in zip(ax, ["pa_deg", "fl_mm", "mt_mm"]):
        a.hist(df[c + "_raw"].dropna(), bins=40, color="#4472c4")
        lo, hi = RANGES[c]
        a.axvline(lo, color="k", ls="--", lw=0.8)
        a.axvline(hi, color="k", ls="--", lw=0.8)
        a.set_title(f"{c} (raw, dashed = official range)")
    plt.tight_layout()
    plt.savefig(fd / f"test_hist_{tag}.png", dpi=90)
    plt.close()
    # 12 overlays: spread over image sizes
    lay = df.merge(pd.read_csv(config.OUTPUTS / "scales.csv")[["image_id", "method"]], on="image_id")
    lay["layout"] = lay.W.astype(str) + "x" + lay.H.astype(str) + lay.method.str.split("(").str[0]
    lay["layout"] = np.where(lay.W < 500, "small", lay.layout)
    pick = lay.groupby("layout").head(2).head(12).image_id.tolist()
    pick += [n for n in df.image_id if n not in pick][: 12 - len(pick)]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for a, name in zip(axes.flat, pick):
        g = np.array(Image.open(config.TEST_DIR / name).convert("L"))
        a.imshow(g, cmap="gray")
        apo, fas = keep[name]
        r = df.set_index("image_id").loc[name]
        if apo is not None:
            xs = np.linspace(0, g.shape[1], 200)
            a.plot(xs, apo["sup_low_lin"](xs), "c-", lw=1)
            a.plot(xs, apo["deep_up_lin"](xs), "y-", lw=1)
        for f in fas:
            (xa, ya), (xb, yb) = f["hit_s"], f["hit_d"]
            a.plot([xa, xb], [ya, yb], "r-", lw=0.8)
        a.set_xlim(0, g.shape[1])
        a.set_ylim(g.shape[0], 0)
        a.set_title(f"{name}  PA {r.pa_deg:.1f}°  FL {r.fl_mm:.0f} mm  MT {r.mt_mm:.1f} mm", fontsize=9)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(fd / f"test_measurements_{tag}.png", dpi=80)
    plt.close()


if __name__ == "__main__":
    main()
