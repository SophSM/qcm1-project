"""Mask -> geometry: aponeurosis curves, fascicle line segments, thickness, angle, length (pixel units).

Conventions: image coordinates (x right, y down). Angles in degrees, measured in the image plane.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


# --------------------------------------------------------------------------- fascicles
def fascicle_segments(mask, min_len=20, max_bend=0.2, prob=None):
    """Skeletonise, cut at branch points, fit a line (PCA) to every piece >= min_len px.

    Returns list of dicts: angle (deg, image plane, in (-90, 90], positive = rising towards the right),
    length (px), c (centre x, y), d (unit direction, dx > 0), and p (mean of `prob` along the skeleton piece,
    only if `prob` is given; run 2 confidence filter).
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return []
    sk = skeletonize(m)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    branch = sk & (nb > 2)
    sk = sk & ~ndi.binary_dilation(branch, iterations=1)
    lab, n = ndi.label(sk, structure=np.ones((3, 3)))
    segs = []
    if n == 0:
        return segs
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        yy, xx = np.nonzero(lab[sl] == i)
        if len(yy) < min_len * 0.7:
            continue
        pts = np.stack([xx + sl[1].start, yy + sl[0].start], 1).astype(np.float64)
        c = pts.mean(0)
        cov = np.cov((pts - c).T)
        ev, evec = np.linalg.eigh(cov)
        d = evec[:, 1]
        if d[0] < 0:
            d = -d
        proj = (pts - c) @ d
        length = proj.max() - proj.min()
        if length < min_len:
            continue
        if np.sqrt(max(ev[0], 0)) / (np.sqrt(ev[1]) + 1e-9) > max_bend:
            continue
        ang = np.degrees(np.arctan2(-d[1], d[0]))
        seg = dict(angle=float(ang), length=float(length), c=c, d=d)
        if prob is not None:
            seg["p"] = float(prob[pts[:, 1].astype(int), pts[:, 0].astype(int)].mean())
        segs.append(seg)
    return segs


def weighted_median(vals, w):
    vals, w = np.asarray(vals, float), np.asarray(w, float)
    if len(vals) == 0:
        return np.nan
    o = np.argsort(vals)
    cw = np.cumsum(w[o])
    return float(vals[o][np.searchsorted(cw, 0.5 * cw[-1])])


def dominant_angle(segs):
    if not segs:
        return np.nan
    return weighted_median([s["angle"] for s in segs], [s["length"] for s in segs])


# --------------------------------------------------------------------------- aponeuroses
APO_PICK = "top"  # default pair rule without fascicle information (see aponeuroses)
class Curve:
    """y = f(x): quadratic inside the observed x-range, straight (tangent) extension outside."""

    def __init__(self, x, y, deg=2):
        self.xmin, self.xmax = float(x.min()), float(x.max())
        deg = deg if (self.xmax - self.xmin) > 50 else 1
        self.p = np.polyfit(x, y, deg)
        self.dp = np.polyder(self.p)

    def __call__(self, x):
        x = np.asarray(x, float)
        xc = np.clip(x, self.xmin, self.xmax)
        return np.polyval(self.p, xc) + np.polyval(self.dp, xc) * (x - xc)

    def slope(self, x):
        return np.polyval(self.dp, np.clip(x, self.xmin, self.xmax))


def _band_profile(cm):
    cols = np.where(cm.any(0))[0]
    sub = cm[:, cols]
    top = sub.argmax(0).astype(float)
    bot = (sub.shape[0] - 1 - sub[::-1].argmax(0)).astype(float)
    return cols.astype(float), top, bot


def aponeurosis_bands(mask, prob=None, min_width_frac=0.12, frag_frac=0.05, min_area_frac=0.15):
    """All band-like structures, sorted top to bottom.

    If `prob` is given, hysteresis thresholding (0.25 / 0.5) keeps weak band parts connected to confident ones.
    Fragments (>= frag_frac of the width) are merged into bands when they lie on the same line (median vertical
    offset of their centre lines < max(8 px, 3 % of height)). Kept: bands spanning >= min_width_frac of the width
    with >= min_area_frac of the largest band's area.
    """
    from skimage.filters import apply_hysteresis_threshold
    m = apply_hysteresis_threshold(prob, 0.25, 0.5) if prob is not None else mask.astype(bool)
    H, W = m.shape
    lab, n = ndi.label(m)
    frags = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None or sl[1].stop - sl[1].start < frag_frac * W:
            continue
        cm = lab == i
        cols, top, bot = _band_profile(cm)
        mid = (top + bot) / 2
        frags.append(dict(area=int(cm.sum()), cols=cols, top=top, bot=bot, mid=mid,
                          line=np.polyfit(cols, mid, 1) if len(cols) > 1 else np.array([0.0, mid[0]])))
    frags.sort(key=lambda f: -f["area"])
    bands = []
    tol = max(8.0, 0.03 * H)
    for f in frags:
        for b in bands:
            if np.median(np.abs(np.polyval(b["line"], f["cols"]) - f["mid"])) < tol:
                for k in ("cols", "top", "bot", "mid"):
                    b[k] = np.concatenate([b[k], f[k]])
                b["area"] += f["area"]
                o = np.argsort(b["cols"])
                for k in ("cols", "top", "bot", "mid"):
                    b[k] = b[k][o]
                b["line"] = np.polyfit(b["cols"], b["mid"], 1)
                break
        else:
            bands.append(dict(f))
    if not bands:
        return []
    amax = max(b["area"] for b in bands)
    bands = [b for b in bands if (b["cols"].max() - b["cols"].min()) >= min_width_frac * W
             and b["area"] >= min_area_frac * amax]
    for b in bands:
        b["cy"] = float(np.mean(b["mid"]))
    return sorted(bands, key=lambda b: b["cy"])


def band_pair(sup, deep):
    out = dict(sup=sup, deep=deep)
    out["sup_low"] = Curve(sup["cols"], sup["bot"])
    out["sup_mid"] = Curve(sup["cols"], sup["mid"])
    out["deep_up"] = Curve(deep["cols"], deep["top"])
    out["deep_mid"] = Curve(deep["cols"], deep["mid"])
    # straight lines (as the expert annotations): used for pennation angle and fascicle length
    out["sup_low_lin"] = Curve(sup["cols"], sup["bot"], deg=1)
    out["deep_up_lin"] = Curve(deep["cols"], deep["top"], deg=1)
    out["x_overlap"] = (max(sup["cols"].min(), deep["cols"].min()), min(sup["cols"].max(), deep["cols"].max()))
    return out


def _gap(a, b, W):
    xs = np.linspace(0, W - 1, 50)
    return float(np.median(np.polyval(b["line"], xs) - np.polyval(a["line"], xs)))


def candidate_pairs(bands, H, W, min_sep_frac=0.08):
    """Adjacent band pairs (upper, next band below that is >= min_sep_frac*H away; closer bands = layers of the
    same aponeurosis)."""
    pairs = []
    for i, a in enumerate(bands):
        for b in bands[i + 1:]:
            if _gap(a, b, W) >= min_sep_frac * H:
                pairs.append((a, b))
                break
    return pairs


def aponeuroses(mask, prob=None, pick=None, fasc=None):
    """Superficial + deep aponeurosis of the measured muscle, or None.

    pick="top": the uppermost valid pair (as DL_Track_US). pick="fasc": the valid adjacent pair with the most
    fascicle pixels (`fasc` = bool mask) between them, i.e. the muscle whose fascicles are visible; falls back to
    "top" when there are no fascicle pixels.
    """
    H, W = mask.shape
    bands = aponeurosis_bands(mask, prob)
    pairs = candidate_pairs(bands, H, W)
    if not pairs:
        return None
    pick = pick or APO_PICK
    if pick == "fasc" and fasc is not None and fasc.any() and len(pairs) > 1:
        fy, fx = np.nonzero(fasc)
        best, bestn = pairs[0], -1
        for a, b in pairs:
            pa, pb = band_pair(a, b)["sup_low"], band_pair(a, b)["deep_up"]
            n = int(((fy > pa(fx)) & (fy < pb(fx))).sum())
            if n > bestn:
                best, bestn = (a, b), n
        return band_pair(*best)
    return band_pair(*pairs[0])


def perp_distance(curve_a, curve_b, x, sx=1.0, sy=1.0, span=None):
    """Shortest distance from point (x, curve_a(x)) to curve_b, in units after scaling x by 1/sx and y by 1/sy."""
    y = float(curve_a(x))
    span = span or 2000
    xs = np.linspace(x - span, x + span, 4001)
    ys = curve_b(xs)
    return float(np.min(np.hypot((xs - x) / sx, (ys - y) / sy)))


def thickness(apo, which="edge", fracs=(0.25, 0.5, 0.75), sx=1.0, sy=1.0):
    """Mean perpendicular distance at 3 positions across the region where both bands are detected."""
    if apo is None:
        return np.nan, []
    a, b = (apo["sup_low"], apo["deep_up"]) if which == "edge" else (apo["sup_mid"], apo["deep_mid"])
    x0, x1 = apo["x_overlap"]
    if x1 - x0 < 10:  # bands don't overlap in x: use the full range covered by either
        x0 = min(apo["sup"]["cols"].min(), apo["deep"]["cols"].min())
        x1 = max(apo["sup"]["cols"].max(), apo["deep"]["cols"].max())
    xs = [x0 + f * (x1 - x0) for f in fracs]
    d = [perp_distance(a, b, x, sx, sy) for x in xs]
    return float(np.mean(d)), xs


def line_curve_intersection(c, d, curve, tmax=20000):
    """Point where line c + t*d meets curve (y = curve(x)), searching both directions. Returns (t, x, y) or None."""
    ts = np.linspace(-tmax, tmax, 40001)
    xs = c[0] + ts * d[0]
    ys = c[1] + ts * d[1]
    f = ys - curve(xs)
    s = np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0]
    if len(s) == 0:
        return None
    k = s[np.argmin(np.abs(ts[s]))]
    # linear refine
    t0, t1, f0, f1 = ts[k], ts[k + 1], f[k], f[k + 1]
    t = t0 - f0 * (t1 - t0) / (f1 - f0 + 1e-12)
    return t, c[0] + t * d[0], c[1] + t * d[1]


def pennation_and_length(segs, apo, sx=1.0, sy=1.0, mt_px=None, max_segs=15):
    """Per fascicle: angle vs the deep aponeurosis (straight-line fit, image plane), and length between the
    aponeurosis edges (straight-line fits) along the extended fascicle line (scaled by sx, sy). Uses only fascicles whose centre lies between the bands.
    Returns (list of dicts with pa, fl, length, ...)."""
    if apo is None or not segs:
        return []
    sup, deep = apo["sup_low_lin"], apo["deep_up_lin"]
    out = []
    for s in sorted(segs, key=lambda s: -s["length"]):
        cx, cy = s["c"]
        if not (sup(cx) - 5 < cy < deep(cx) + 5):
            continue
        hit_d = line_curve_intersection(s["c"], s["d"], deep)
        hit_s = line_curve_intersection(s["c"], s["d"], sup)
        if hit_d is None or hit_s is None:
            continue
        xd = hit_d[1]
        apo_ang = np.degrees(np.arctan(-deep.slope(xd)))  # rising to the right = positive
        pa = abs(s["angle"] - apo_ang)
        pa = min(pa, 180 - pa)
        fl = float(np.hypot((hit_d[1] - hit_s[1]) / sx, (hit_d[2] - hit_s[2]) / sy))
        out.append(dict(pa=float(pa), fl=fl, length=s["length"], seg=s, hit_s=hit_s[1:], hit_d=hit_d[1:]))
        if len(out) >= max_segs:
            break
    return out
