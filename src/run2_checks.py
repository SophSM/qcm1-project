"""Run 2 (TRAINING_PLAN_2 §3.4-3.5): build v6 and run the test-set sanity checks (report only, no tuning).

    python src/run2_checks.py --v4 <submission_v4 file> --v5 <submission_v5 file>

v6 = per-image mean of v2 and v3 for PA and FL; MT = v3 exactly (plan §1.3). Then for v3 / v4 / v6:
FL distribution (figure), values clipped to 30 / 200 mm, within-clip stability SD, |v4 - v3| per variable,
the 10 largest FL changes (+ overlay figure of the 4 largest from distinct clips), and FL*sin(PA)/MT (should be ~1).
Outputs: outputs/submission_v6_avg_v2_v3.csv, outputs/run2_checks.json,
         outputs/figures/fl_test_hist_v4.png, outputs/figures/fl_changes_v4.png
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from measure import RULES, check_submission, load_prob, measure_image  # noqa: E402

O = config.OUTPUTS
V2, V3 = "submission_v2_apoAB_fascA.csv", "submission_v3_alldata.csv"
V6 = "submission_v6_avg_v2_v3.csv"


def read(f):
    return pd.read_csv(O / f).set_index("image_id")


def meas(f):
    return pd.read_csv(O / ("measurements_" + os.path.splitext(f)[0] + ".csv")).set_index("image_id")


def make_v6():
    a, b = read(V2), read(V3)
    v6 = b.copy()
    v6[["pa_deg", "fl_mm"]] = (a.loc[b.index, ["pa_deg", "fl_mm"]] + b[["pa_deg", "fl_mm"]]) / 2
    v6 = v6.round(3).reset_index()
    v6.to_csv(O / V6, index=False, float_format="%.3f")
    check_submission(O / V6, pd.read_csv(config.SAMPLE_SUBMISSION, sep=";", encoding="utf-8-sig"))
    assert (read(V6).mt_mm == b.mt_mm).all()
    print("v6 MT identical to v3")


def stability(sub, groups):
    d = sub.join(groups)
    g = d.groupby("group").filter(lambda x: len(x) >= 2).groupby("group")[["pa_deg", "fl_mm", "mt_mm"]].std()
    return {c: dict(mean=round(float(g[c].mean()), 3), median=round(float(g[c].median()), 3)) for c in g}, len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4", required=True)
    ap.add_argument("--v5", required=True)
    ap.add_argument("--rule", default="k10dir12")
    args = ap.parse_args()
    make_v6()
    subs = {"v2": read(V2), "v3": read(V3), "v4": read(args.v4), "v5": read(args.v5), "v6": read(V6)}
    m3, m4 = meas(V3), meas(args.v4)
    groups = m3[["group"]]
    res = {"files": {"v4": args.v4, "v5": args.v5, "v6": V6}}
    for k, s in subs.items():
        st, ng = stability(s, groups)
        ratio = s.fl_mm * np.sin(np.radians(s.pa_deg)) / s.mt_mm
        res[k] = dict(fl_median=round(float(s.fl_mm.median()), 2), fl_sd=round(float(s.fl_mm.std()), 2),
                      fl_min=float(s.fl_mm.min()), fl_max=float(s.fl_mm.max()),
                      n_fl_30=int((s.fl_mm <= 30).sum()), n_fl_200=int((s.fl_mm >= 200).sum()),
                      stability_sd=st, n_groups=ng,
                      ratio_flsinpa_mt=dict(median=round(float(ratio.median()), 3),
                                            q10=round(float(ratio.quantile(0.1)), 3),
                                            q90=round(float(ratio.quantile(0.9)), 3)))
    for a, b in (("v4", "v3"), ("v5", "v2"), ("v6", "v3")):
        d = (subs[a] - subs[b].loc[subs[a].index]).abs()
        res[f"diff_{a}_{b}"] = {c: dict(mean=round(float(d[c].mean()), 3), max=round(float(d[c].max()), 3))
                                for c in d}
    # PA identical for rules with pa_from=run1
    res["pa_v4_equals_v3"] = bool((subs["v4"].pa_deg == subs["v3"].pa_deg).all())
    dfl = (subs["v4"].fl_mm - subs["v3"].fl_mm).rename("dFL")
    top = pd.concat([subs["v3"].fl_mm.rename("fl_v3"), subs["v4"].fl_mm.rename("fl_v4"), dfl,
                     m3.n_fasc.rename("n_fasc_v3"), m4.n_fasc.rename("n_fasc_v4"),
                     (m3.W.astype(str) + "x" + m3.H.astype(str)).rename("size")], axis=1)
    top = top.reindex(dfl.abs().sort_values(ascending=False).index).head(10)
    res["top10_fl_change"] = top.round(2).reset_index().to_dict("records")
    res["n_fl_changed"] = int((dfl.abs() > 0.0005).sum())
    res["n_fl_changed_gt5mm"] = int((dfl.abs() > 5).sum())
    (O / "run2_checks.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    order = dfl.abs().sort_values(ascending=False).index
    pick = groups.loc[order].drop_duplicates("group").index[:4].tolist()  # 4 largest changes, distinct clips
    figures(m3, m4, subs, pick, args.rule)


def figures(m3, m4, subs, pick, rule):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = O / "figures"
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    bins = np.linspace(0, 260, 53)
    for k, col in (("v3", "#8c8c8c"), ("v4", "#4472c4")):
        raw = (m3 if k == "v3" else m4).fl_mm_raw.dropna()
        ax[0].hist(raw, bins=bins, histtype="step", lw=1.6, color=col, label=f"{k} raw (n={len(raw)})")
    ax[0].axvline(30, color="k", ls="--", lw=0.8)
    ax[0].axvline(200, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel("FL (mm), before clipping; dashed = official range")
    ax[0].legend(frameon=False)
    d = subs["v4"].fl_mm - subs["v3"].fl_mm
    ax[1].hist(d, bins=41, color="#4472c4")
    ax[1].set_xlabel("FL v4 - v3 (mm), per test image")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(fd / "fl_test_hist_v4.png", dpi=100)
    plt.close()
    pdir = O / "pred_masks" / "test_v3"
    boxes = pd.read_csv(pdir / "boxes.csv").set_index("file")
    scales = pd.read_csv(O / "scales.csv").set_index("image_id")
    fig, axes = plt.subplots(1, len(pick), figsize=(5.5 * len(pick), 4.5))
    for a, n in zip(np.atleast_1d(axes), pick):
        g = np.array(Image.open(config.TEST_DIR / n).convert("L"))
        a.imshow(g, cmap="gray")
        b, sc = boxes.loc[n], scales.loc[n]
        ap_, fp_ = load_prob(pdir / "apo", n), load_prob(pdir / "fasc", n)
        for rn, col, lw in (("run1", "r", 1.4), (rule, "#00b0f0", 0.8)):
            r, apo, fas = measure_image(ap_, fp_, (b.x0, b.y0, b.x1, b.y1), sc.sx, sc.sy, RULES[rn])
            if apo is not None and rn == "run1":
                xs = np.linspace(0, g.shape[1], 200)
                a.plot(xs, apo["sup_low_lin"](xs), "c-", lw=1)
                a.plot(xs, apo["deep_up_lin"](xs), "y-", lw=1)
            for f in fas:
                (xa, ya), (xb, yb) = f["hit_s"], f["hit_d"]
                a.plot([xa, xb], [ya, yb], color=col, lw=lw)
        a.set_xlim(0, g.shape[1])
        a.set_ylim(g.shape[0], 0)
        a.set_title(f"{n}: FL v3 {subs['v3'].fl_mm[n]:.0f} -> v4 {subs['v4'].fl_mm[n]:.0f} mm\n"
                    f"red = run-1 fascicles (7), blue = {rule} (10)", fontsize=9)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(fd / "fl_changes_v4.png", dpi=80)
    plt.close()


if __name__ == "__main__":
    main()
