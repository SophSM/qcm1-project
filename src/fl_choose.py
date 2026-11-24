"""Run 2 (TRAINING_PLAN_2 §3.3): pick the FL/PA rule from outputs/fl_tuning.csv.

    python src/fl_choose.py     (light: reads CSV/NPZ only)

Procedure (conservative, rule §1.4 of the plan):
  1. one family at a time (all other parameters at run-1 values): table per family;
  2. best rule with <= 3 parameters changed from run 1;
  3. simplest rule (fewest changes, then lowest score) whose score is within 3 % of that best;
  4. if its PA error is worse than run 1, use run-1 PA (pa_from = run1) and change only FL;
  5. a gain < 3 % over run 1 counts as noise -> no FL improvement (plan §5).
Also: paired bootstrap over validation clips ("unit" in split_fasc.csv) of the score difference vs run 1.
Writes outputs/fl_choice.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
from fl_tune import FAMILIES, FAMILY_LETTER  # noqa: E402
from measure import RUN1  # noqa: E402

TOL = 0.03


def is_val(x, v):
    return (v is None and pd.isna(x)) or (v is not None and not pd.isna(x) and x == v)


def boot_diff(raw, a, b, n=2000, seed=0):
    """Clip-level paired bootstrap of score(a) - score(b): mean and 95 % interval."""
    names = list(raw["names"])
    ia, ib = names.index(a), names.index(b)
    unit = raw["unit"]
    pa, fl = raw["pa_err"], raw["fl_err_mm"]
    us = np.unique(unit)
    idx = {u: np.where(unit == u)[0] for u in us}
    rng = np.random.default_rng(seed)

    def sc(i, rows):
        f = fl[i, rows]
        return 0.5 * np.nanmean(f) / 10 + 0.5 * pa[i, rows].mean() / 4

    d = []
    for _ in range(n):
        rows = np.concatenate([idx[u] for u in rng.choice(us, len(us))])
        d.append(sc(ia, rows) - sc(ib, rows))
    return float(np.mean(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], int(len(us))


def main():
    df = pd.read_csv(config.OUTPUTS / "fl_tuning.csv")
    raw = np.load(config.OUTPUTS / "fl_tune_raw.npz", allow_pickle=True)
    run1 = df[df.n_changes == 0].iloc[0]
    print(f"run 1: score {run1.score:.4f}  FL err {run1.fl_err_mm_mean:.2f} mm  PA err {run1.pa_err_mean:.3f} deg")
    fam_rows = []
    for k, opts in FAMILIES.items():
        for v in opts:
            m = df.pa_from.isna() & df[k].apply(lambda x: is_val(x, v))
            m &= df.n_changes == (0 if v == RUN1[k] else 1)
            r = df[m].iloc[0]
            fam_rows.append(dict(family=FAMILY_LETTER[k] + ". " + k, option="none" if v is None else v,
                                 score=r.score, rel=r.score / run1.score - 1, fl_err_mm=r.fl_err_mm_mean,
                                 pa_err=r.pa_err_mean, n_missing=r.n_missing))
    fam = pd.DataFrame(fam_rows)
    print(fam.round(4).to_string(index=False))
    fam.to_csv(config.OUTPUTS / "fl_tuning_families.csv", index=False)

    elig = df[df.n_changes <= 3]
    best = elig.iloc[elig.score.argmin()]
    near = elig[elig.score <= best.score * (1 + TOL)].sort_values(["n_changes", "score"])
    chosen = near.iloc[0]
    print("best <=3 changes:", best.rule, round(best.score, 4))
    print("simplest within 3 %:", chosen.rule, round(chosen.score, 4), "changes", chosen.n_changes)
    pa_note = ""
    if chosen.pa_err_mean > run1.pa_err_mean and pd.isna(chosen.pa_from):
        alt = df[df.rule == chosen.rule + "_paRun1"]
        if len(alt):
            pa_note = f"PA error worse than run 1 ({chosen.pa_err_mean:.3f} > {run1.pa_err_mean:.3f}) -> run-1 PA"
            chosen = alt.iloc[0]
    runner = near.iloc[1] if len(near) > 1 else elig.sort_values("score").iloc[1]
    gain = 1 - chosen.score / run1.score
    d, ci, nu = boot_diff(raw, chosen.rule, run1.rule)
    out = dict(run1=run1.rule, run1_score=run1.score, best=best.rule, best_score=best.score,
               chosen=chosen.rule, chosen_score=chosen.score, chosen_changes=int(chosen.n_changes),
               chosen_params={k: (None if pd.isna(chosen[k]) else chosen[k]) for k in list(FAMILIES) + ["pa_from"]},
               runner_up=runner.rule, runner_up_score=runner.score, gain_vs_run1=gain,
               improves=bool(gain >= TOL), pa_note=pa_note,
               boot_diff_vs_run1=d, boot_ci95=ci, n_clips=nu,
               chosen_fl_err_mm=chosen.fl_err_mm_mean, chosen_pa_err=chosen.pa_err_mean,
               run1_fl_err_mm=run1.fl_err_mm_mean, run1_pa_err=run1.pa_err_mean)
    for k, v in out["chosen_params"].items():
        if isinstance(v, (np.integer, np.floating)):
            out["chosen_params"][k] = v.item()
    (config.OUTPUTS / "fl_choice.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
