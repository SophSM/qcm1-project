"""Pixel -> millimetre scale from the on-screen rulers (content-based, no filenames).

Ruler types found in the test screenshots (see outputs/DECISIONS.md / REPORT.md):
  * left ruler with short tick marks, outside the scan (left of it):
      - minor + major ticks (every 2nd tick longer)       -> minor tick = 5 mm  (cm-labelled ruler, "P" device)
      - one tick length only                              -> tick = 10 mm      (Telemed, labels 0..50 mm)
    gives the vertical (depth) scale sy.
  * bottom-edge tick row (Telemed lateral ruler, also on cropped Telemed images) -> tick = 10 mm, horizontal sx.
  * right-hand tick column (Siemens): tick spacing depends on depth (2 or 5 mm ...). No usable label without OCR,
    so the tick size s in {1, 2, 5, 10} mm is chosen so that the scan width matches the probe's field of view
    (SIEMENS_FOV_MM, measured on images where the depth label could be read).
If only one axis is known, the other is assumed equal (isotropic). If nothing is found: median of the confidently
measured images with the same image size, else the global median (confidence 'low').

Usage: python src/scale.py  (CPU job; reads outputs/crops.csv, writes outputs/scales.csv)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from crop import load_rgb  # noqa: E402
from data import list_images  # noqa: E402

SIEMENS_FOV_MM = 57.5
TICK_CANDIDATES_MM = (1, 2, 5, 10)


def runs(b, max_len=4):
    """Centres of runs of True in 1-D bool array whose length <= max_len (short tick marks)."""
    idx = np.where(b)[0]
    out = []
    if len(idx) == 0:
        return out
    s = p = idx[0]
    for i in list(idx[1:]) + [None]:
        if i is None or i - p > 1:
            if p - s + 1 <= max_len:
                out.append((s + p) / 2)
            if i is None:
                break
            s = i
        p = i
    return out


def regular(c, min_n=4, min_period=20, tol=0.03):
    """Longest equally spaced lattice c0 + k*p hit by the centres (extra, non-lattice marks are ignored).
    Returns (period, n, first, last) or None."""
    c = np.asarray(c, float)
    if len(c) < min_n:
        return None
    best = None
    for i in range(len(c) - 1):
        for j in range(i + 1, len(c)):
            p = c[j] - c[i]
            if p < min_period:
                continue
            if best is not None and (c[-1] - c[i]) / p + 1 < best[1]:
                break
            hits, k = [c[i]], 1
            while True:
                t = c[i] + k * p
                d = np.abs(c - t)
                m = d.argmin()
                if d[m] > max(1.5, tol * p):
                    break
                hits.append(c[m])
                p = (c[m] - c[i]) / k  # refine
                k += 1
            n = len(hits)
            if n >= min_n and (best is None or n > best[1]):
                best = ((hits[-1] - hits[0]) / (n - 1), n, hits[0], hits[-1])
    return best


def column_ticks(g, xs, thr=100, **kw):
    """Best regular tick sequence among columns xs (ticks = short bright vertical runs)."""
    best = None
    for x in xs:
        r = regular(runs(g[:, x] > thr), **kw)
        if r and (best is None or r[1] > best[1][1]):
            best = (x, r)
    return best


def row_ticks(g, ys, thr=100, **kw):
    best = None
    for y in ys:
        r = regular(runs(g[y] > thr), **kw)
        if r and (best is None or r[1] > best[1][1]):
            best = (y, r)
    return best


def detect_scale(rgb, box):
    g = rgb.astype(np.float32).mean(2)
    H, W = g.shape
    x0, y0, x1, y1 = box
    ch, cw = y1 - y0, x1 - x0
    res = dict(sx=np.nan, sy=np.nan, method="", confidence="", tick_px=np.nan, tick_mm=np.nan)

    # 1) left ruler (vertical / depth scale)
    left = column_ticks(g, range(0, max(0, x0 - 3)), min_n=5, min_period=25)
    if left and (left[1][3] - left[1][2]) > 0.4 * ch:
        x, (per, n, a, b) = left
        # minor/major: a neighbouring column carrying every 2nd tick only
        mm = 10.0
        for xx in range(max(0, x - 8), min(W, x + 9)):
            r = regular(runs(g[:, xx] > 100), min_n=3, min_period=25)
            if r and abs(r[0] - 2 * per) < max(2, 0.03 * per * 2):
                mm = 5.0
                break
        res.update(sy=per / mm, tick_px=per, tick_mm=mm, method=f"left ruler ({n} ticks, {mm:g} mm)",
                   confidence="high")

    # 2) bottom-edge ruler (horizontal scale), only below the scan
    bot = row_ticks(g, range(max(0, H - 14), H), min_n=4, min_period=25)
    if bot is None:  # ticks touched by bright speckle: vote over 4 consecutive rows
        best = None
        for y in range(max(0, H - 14), H - 3):
            b = (g[y:y + 4] > 100).sum(0) >= 3
            r = regular(runs(b, max_len=5), min_n=4, min_period=25)
            if r and (best is None or r[1] > best[1][1]):
                best = (y, r)
        bot = best
    if bot and not np.isnan(res["sy"]) and not (0.8 < (bot[1][0] / 10.0) / res["sy"] < 1.25):
        bot = None  # disagrees strongly with the depth ruler -> spurious (speckle) match
    if bot and (bot[1][3] - bot[1][2]) > 0.4 * cw:
        y, (per, n, a, b) = bot
        res["sx"] = per / 10.0
        res["method"] = (res["method"] + " + " if res["method"] else "") + f"bottom ruler ({n} ticks, 10 mm)"
        res["confidence"] = res["confidence"] or "high"
        if np.isnan(res["tick_px"]):
            res.update(tick_px=per, tick_mm=10.0)

    # 3) right tick column (Siemens), only if nothing else found
    if np.isnan(res["sx"]) and np.isnan(res["sy"]):
        right = column_ticks(g, range(min(W - 1, x1 + 40), W), thr=60, min_n=5, min_period=12)
        if right and (right[1][3] - right[1][2]) > 0.6 * ch:
            x, (per, n, a, b) = right
            fovs = {s: cw / (per / s) for s in TICK_CANDIDATES_MM}
            s = min(fovs, key=lambda k: abs(np.log(fovs[k] / SIEMENS_FOV_MM)))
            ok = abs(np.log(fovs[s] / SIEMENS_FOV_MM)) < np.log(1.25)
            res.update(sy=per / s, tick_px=per, tick_mm=s, confidence="medium" if ok else "low",
                       method=f"right ticks ({n} ticks, {s} mm by FOV {fovs[s]:.1f} mm)")

    res["has_x"] = not np.isnan(res["sx"])
    if np.isnan(res["sx"]):
        res["sx"] = res["sy"]
    if np.isnan(res["sy"]):
        res["sy"] = res["sx"]
    return res


def main():
    crops = pd.read_csv(config.OUTPUTS / "crops.csv")
    crops = crops[crops.set == "test"].set_index("file")
    rows = []
    for f in list_images(config.TEST_DIR):
        r = crops.loc[f]
        rgb = load_rgb(config.TEST_DIR / f)
        s = detect_scale(rgb, (r.x0, r.y0, r.x1, r.y1))
        rows.append(dict(image_id=f, W=r.W, H=r.H, x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1, **s))
    df = pd.DataFrame(rows)
    # depth ruler only, but same-layout images (same size, same depth scale) also show a lateral ruler:
    # take their horizontal scale (handles anisotropic screenshots)
    for i in df.index[(df.method.str.startswith("left ruler")) & ~df.has_x]:
        same = df[df.has_x & df.method.str.startswith("left ruler") & (df.W == df.at[i, "W"]) &
                  (df.H == df.at[i, "H"]) & ((df.sy - df.at[i, "sy"]).abs() < 0.01 * df.at[i, "sy"])]
        if len(same):
            df.loc[i, "sx"] = same.sx.median()
            df.loc[i, "method"] += f" + sx from {len(same)} same-layout images"
    # fallback for images without a detected ruler: same image size + similar crop, else global
    ok = df.confidence.isin(["high", "medium"])
    for i in df.index[~ok | df.sx.isna()]:
        same = df[ok & (df.W == df.at[i, "W"]) & (df.H == df.at[i, "H"])]
        src = same if len(same) else df[ok]
        df.loc[i, ["sx", "sy"]] = src.sx.median(), src.sy.median()
        df.loc[i, "method"] = "fallback: median of " + ("same image size" if len(same) else "all confident images")
        tight = len(same) >= 3 and same.sx.std() / same.sx.median() < 0.02
        df.loc[i, "confidence"] = "medium" if tight else "low"
    df["aniso"] = df.sx / df.sy
    df.to_csv(config.OUTPUTS / "scales.csv", index=False)
    print(df.groupby(["W", "H", "method"]).agg(n=("image_id", "size"), sx=("sx", "median"), sy=("sy", "median"))
          .to_string())
    print(df.confidence.value_counts().to_string())


if __name__ == "__main__":
    main()
