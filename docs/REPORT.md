> **Note for this repository.** Paths such as `outputs/...` refer to the working folder of the run, which is not
> part of the repository. Only figures that are plain plots are included in `docs/figures/`. Figures that show
> competition images (crop checks, validation examples, test overlays) are left out because the data licence does
> not allow sharing the data, so those links do not resolve here.

# UMUD challenge: training run report (13 Nov 2026)

> **Run 2 (14 Nov 2026, fascicle length): see §10.** It found no reliable FL improvement. Upload
> `submission_v6_avg_v2_v3.csv` first. The upload advice in the summary below is from run 1.

## Summary (5 sentences)
1. Two U-Net models were trained: one finds the two **aponeuroses** (the bright bands around the muscle) and
   matches the expert masks well (validation Dice 0.83); the other finds **fascicles** (the slanted fibre lines)
   with Dice 0.58, which is good for this task because the labels mark only some of the visible fascicles.
2. On held-out validation images, **muscle thickness** from our masks differs from thickness measured on the expert
   masks by **0.55 mm on average (median 0.18 mm)**, and the **fascicle direction** differs by **0.75° on
   average**: both are well inside the raters' own agreement limits (1 mm, 4°).
3. The **millimetre scale** is read from the on-screen rulers of every test image (217 of 309 high confidence,
   92 medium); for the Siemens layout it was cross-checked against the printed depth label on six images (all
   within ~2 %).
4. **Fascicle length** is the least certain value: it is extrapolated from a line beyond the image (as the experts
   do), so small angle errors become large length errors, and there is no ground truth to validate it.
5. The test set comes from a new scanner, a new muscle and new participants, and our validation only measures
   agreement with *masks*, not with the experts' *numbers*, so the true leaderboard error will be larger than the
   numbers above. The Kaggle score is the real test.

**Upload first:** `outputs/submission.csv` (= `submission_v2_apoAB_fascA.csv`).
**The two final picks:** `submission_v2_apoAB_fascA.csv` (validated models) and `submission_v3_alldata.csv` (same
recipe retrained on train + validation images). `submission_v1_baseline.csv` is the early safety version (weaker
models, older geometry); don't pick it as final.

---

## 1. What was built (plain words)
1. **Crop** each screenshot to the actual ultrasound picture (text, rulers and colour bars removed).
   [crop check](figures/crop_check.png)
2. **Two U-Nets** (ResNet34 encoder pretrained on ImageNet). Aponeurosis model: whole crop at 512×512.
   Fascicle model: higher resolution (crop scaled to 768 px height), trained on 512×512 patches, predicted with a
   sliding window. Fascicle masks thickened to ~5 px, because we need their direction, not their exact edges.
3. **Geometry**, copying the experts' protocol (COMPETITION_INFO.md):
   - **MT**: perpendicular distance between the lower edge of the upper band and the upper edge of the lower band,
     at 25/50/75 % of the width, averaged.
   - **PA**: angle between each fascicle line and the deep aponeurosis (straight-line fit); median of the 7 longest
     fascicles between the bands.
   - **FL**: the fascicle line extended until it meets both aponeuroses (also beyond the image); median length.
4. **Pixel → mm** from the ruler tick marks on each image.
5. Failed measurements → median of the confident images. Values outside the official ranges → clipped to the range.

All decisions, with reasons: [`DECISIONS.md`](DECISIONS.md).

## 2. Validation split
Training images are video frames, and half of the fascicle set (1,385 images) are exact duplicates. Duplicates,
near-duplicates (thumbnail correlation > 0.99) and runs of consecutive similar frames ("clips") always stay on the
same side of the split. Result: aponeurosis 890 train / 158 val; fascicle 2,341 train / 420 val.
([neighbouring frames](figures/neighbour_frames_fasc.png))

## 3. Segmentation results (validation, original resolution; models used in v2)
| | Aponeurosis (A+B ensemble) | Fascicle (variant A) |
|---|---|---|
| Dice, mean ± sd (median) | 0.83 ± 0.12 (0.86) | 0.58 ± 0.09 (0.58) |
| IoU, mean (median) | 0.72 (0.75) | 0.42 (0.41) |
| Measurement check | thickness: mean error **6.7 px / 0.55 mm**, median **2.4 px / 0.18 mm**, correlation 0.96 | dominant fascicle angle: mean error **0.75°**, median **0.54°**, 90th percentile 1.5° |

**How to read this.** Dice = overlap between our mask and the expert mask (1 = perfect). 0.83 for bands is good.
0.58 for fascicles looks low but is expected: the expert masks mark only *some* fascicles as 1–2 px lines, so the
model is "penalised" for finding unlabelled ones (e.g. image_0231 has an empty label, yet the model draws plausible
fascicles, giving Dice 0). What matters is the **direction**, and that agrees to within ~0.5–0.8°.
The thickness error is mostly tiny, with a few outliers where the expert mask has three bands (two stacked muscles)
and the "which pair" choice differs.
Figures: [aponeurosis examples](figures/val_examples_apo_final.png),
[fascicle examples](figures/val_examples_fasc_final.png) (4 random + 4 worst each).
Training curves: [apo A](figures/train_apo_A.png), [apo B](figures/train_apo_B.png),
[fasc A](figures/train_fasc_A.png).

**Proxy UMUD score (validation)** = mean(MT error / 1 mm, angle error / 4°) = (0.55 + 0.19) / 2 = **0.37**
(v1 baseline: 0.47). FL has no proxy. This is optimistic. It compares with mask-derived values on the same kind of
images, not with the experts' measurements on the shifted test set.

### Model versions
| Version | Aponeurosis | Fascicle | Val apo Dice / MT err (mm) | Val fasc Dice / angle err |
|---|---|---|---|---|
| v1 baseline | B, 10 epochs | B, 10 epochs | 0.80 / 0.74* | 0.56 / 0.81° |
| **v2** | A + B ensemble, 46 epochs | A, best epoch 19 | **0.83 / 0.55** | **0.58 / 0.75°** |
| v3 all data | same, retrained on train+val | same | n/a (no held-out data) | n/a |

\*v1 with the older geometry. Aponeurosis A alone: 0.83 / 0.82 mm; B alone: 0.82 / 0.70 mm. The ensemble beat both.

## 4. Enhancement experiment (fascicle model, 20 epochs, same split)
| Run | Input | Val Dice | Angle error (mean) |
|---|---|---|---|
| **A** | grey only | **0.579** | **0.75°** |
| B | grey + local contrast (CLAHE) + line highlighting (ridge filter) | 0.569 | 0.76° |
| C | B + mild grain removal (bilateral) | 0.573 | 0.80° |
| D | B + stronger grain removal (non-local means) | 0.571 | 0.83° |

**Chosen: A (plain grey).** The enhancements did not help, and stronger grain removal hurt the angle slightly,
as feared. For the aponeuroses, A and B were both trained in full; their ensemble is used because it beat each one
alone on validation thickness error. A full variant-B fascicle model was also trained: ensembling it with A gave no
gain (Dice 0.587 vs 0.585, angle 0.76° vs 0.75°), so A is used alone.
[preprocessing examples](figures/preprocess_examples.png)

## 5. Pixel → millimetre scale
Nothing in the data gives the pixel size, so it is read from the on-screen rulers (`src/scale.py`, idea as in
DL_Track_US's scaling-bar calibration, made automatic). Per test image: `outputs/scales.csv`.

| Test layout (detected by content) | Images | px/mm (median) | How | Confidence |
|---|---|---|---|---|
| Siemens screenshot 1200×800 | 91 | 15.2 (8.7–16.0) | right tick column. Tick = 2 or 5 mm depending on depth, chosen so the scan width matches the probe's 57.5 mm field of view. **Checked against the printed depth label on 6 images (3–6.5 cm): all within ~2 %** | medium |
| Telemed screenshot 1200×800 | 90 | 14.8 vertical, 13.45 horizontal | left 0–50 mm ruler + bottom ruler (10 mm ticks). **This layout is stretched unequally** (non-square pixels), so x and y scales are used separately | high |
| "P" device (likely Philips Lumify, new scanner) 1200×800 PNG | 58 | 15.1 (12.1–20.1) | left cm ruler, minor ticks 5 mm. Consistent span for all depths | high |
| Telemed screen 1088×644 | 50 | 12.6 | left + bottom rulers agree (12.61 / 12.61) | high |
| Telemed scan crops 1069/959×853 | 12 | 16.7 | bottom ruler (10 mm). 1 image by same-size fallback | high / medium |
| Telemed scan crops ~465×513 | 8 | 7.8 | bottom ruler (10 mm) | high |

Confidence over all 309: **high 217, medium 92, low 0**. Main residual risks: (1) the assumption that bottom-ruler
ticks are 10 mm on the cropped Telemed images (consistent with a 60 mm probe field of view, but no label to read);
(2) for anisotropic screenshots we don't know whether the experts used separate x/y scales.
PA doesn't depend on the scale.

## 6. Test predictions
Distribution: [histograms](figures/test_hist_submission_v2_apoAB_fascA.png). Overlays for 12 test images of
all layouts: [test measurements](figures/test_measurements_submission_v2_apoAB_fascA.png).

| | min | median | max | official range |
|---|---|---|---|---|
| PA (°) | 7.6 | 14.2 | 35.4 | 5–45 |
| FL (mm) | 30.0 | 88.4 | 160.5 | 30–200 |
| MT (mm) | 12.1 | 20.7 | 34.0 | 10–50 |

**Flagged / replaced (v2):** 5 images (one 5-frame clip, Telemed 1200×800, IMG_00186–190) had no detectable deep
aponeurosis → all three values = median of the other images. 3 FL values < 30 mm (new "P" device) → clipped to 30.
No other replacements. (In v1, 24–38 values per variable had to be replaced.)

Per layout (median): Siemens PA 15.1° / FL 85 mm / MT 20.7 mm; Telemed-1200 12.2° / 102 / 21.4;
Telemed-1088 12.9° / 112 / 25.7; "P" device 20.1° / 59 / 19.3; Telemed crops 18–19° / 64–66 / 16–17.

**New-scanner check ("P" device, likely Philips Lumify, 58 images):** in v1 the aponeurosis model missed the deep
band on 13 of these images. A training augmentation that imitates their narrow scan window, plus geometry that joins
band fragments, fixed all of them (0 missing in v2). Their PA values are higher and FL shorter than the other
layouts, which fits a different muscle (rectus femoris / gastrocnemius) but cannot be verified.

**Temporal stability:** 35 groups of near-identical test frames found by image content (29 of 5 frames, 6 of 2).
Mean within-group SD: **PA 0.63°, FL 3.3 mm, MT 0.31 mm** (medians 0.27° / 1.6 mm / 0.01 mm). One group has an
MT SD of 4.2 mm, where one frame picked a different band pair. No smoothing across frames was applied.

## 7. Known weaknesses
1. **FL is the weakest value**: it is extrapolated, highly sensitive to angle (FL ≈ MT / sin PA), and unvalidated.
2. **Which muscle / which band pair** is measured when the image shows two muscles: we take the adjacent pair that
   contains the most detected fascicles (else the uppermost pair). The experts may choose differently, e.g. for
   the new rectus femoris images.
3. **Validation ≠ test**: our proxy uses the training masks, not expert numbers, and test images come from a new
   scanner, muscle and population. Which 3 fascicles the experts picked is unknown; we use the median of the 7
   longest.
4. **Scale assumptions** (see §5): Siemens tick size via the field of view (verified on 6 images); 10 mm ticks on the
   cropped Telemed images; anisotropy handling.
5. One clip (5 images) gets median values only.

## 8. Three most promising next steps
1. **Use the leaderboard sparingly for global choices**: e.g. compare a PA/FL variant using all fascicles vs the
   7 longest, or inner-edge vs centre-line MT. Global settings only, never per-image.
2. **Read the depth label (OCR)** on Siemens and "P" screens as a second, independent scale estimate, and flag
   disagreements.
3. **Better fascicle-length estimate**: fit one common fascicle direction per image (robust line field) instead of
   per-segment lines, and test FL = MT / sin(PA) as a more stable alternative. Possibly add a pseudo-labelled pass
   on the test-like "P" device images.

## 9. How to re-run (exact commands, from the project root on the cluster)
```bash
module load anaconda3/3.11 && source activate /home/sofias/software/envs/tools
sbatch slurm/prep.slurm                         # crops.csv, split_*.csv, input cache (CPU, ~25 min)
# models used in v2
sbatch --job-name=apo_A  slurm/train.slurm --task apo  --variant A --epochs 60 --patience 15 --name apo_A
sbatch --job-name=apo_B  slurm/train.slurm --task apo  --variant B --epochs 60 --patience 15 --name apo_B
sbatch --job-name=fasc_A slurm/train.slurm --task fasc --variant A --epochs 60 --patience 12 --name fasc_A
# test masks + validation masks + validation metrics + scales + measurements + submission
sbatch --export=ALL,APO="models/apo_A.pt models/apo_B.pt",FASC=models/fasc_A.pt,TAG=v2,OUT=submission_v2_apoAB_fascA.csv slurm/predict.slurm
# v3 (all data, fixed epochs)
sbatch slurm/train.slurm --task apo  --variant A --epochs 46 --all_data --save_last --name apo_A_all
sbatch slurm/train.slurm --task apo  --variant B --epochs 46 --all_data --save_last --name apo_B_all
sbatch slurm/train.slurm --task fasc --variant A --epochs 19 --all_data --save_last --name fasc_A_all
sbatch --export=ALL,APO="models/apo_A_all.pt models/apo_B_all.pt",FASC=models/fasc_A_all.pt,TAG=v3,OUT=submission_v3_alldata.csv slurm/predict.slurm
```
Inference only (weights exist): the last line of either block. The pipeline never uses file names or order at
prediction time: crop, scale, band choice and measurements come from each image's content, and the median fallback
is a global rule. Seeds fixed (`config.SEED = 42`). Note: the cuDNN kernels are not set to deterministic mode, so
retraining gives very close but not bit-identical weights; inference from the saved weights is deterministic.

**Weights:** `models/apo_A.pt`, `models/apo_B.pt`, `models/fasc_A.pt` (v2); `models/*_all.pt` (v3);
`models/apo_v1.pt`, `models/fasc_v1.pt` (v1); experiments `models/exp_fasc_[A-D].pt`, `models/fasc_B.pt`.
The plan's names `models/apo_best.pt` / `models/fasc_best.pt` are copies of the v2 fascicle model and the
aponeurosis A model (the v2 aponeurosis prediction is the A+B ensemble).

## 10. Run 2: fascicle length (14 Nov 2026, `TRAINING_PLAN_2.md`)

**In short:** we tried 1,944 ways of turning the existing fascicle masks into FL (and PA) values, without
retraining. **None was reliably better than the run-1 rule.** The best one was 2.3 % better on our validation proxy,
which is inside the noise (needed: ≥ 3 %; a bootstrap over validation clips gives a 95 % interval of about ±2 %
around zero). So the plan's fallback applies: **upload `submission_v6_avg_v2_v3.csv` first** (the average of the
two scored submissions, which needs no tuning). v4/v5 contain the best-but-not-significant rule, for an optional
leaderboard test.

### 10.1 The FL proxy and its limits
FL has no ground truth, but the fascicle validation images (420, never trained on by the v2 models) have expert
fascicle masks:
- **Reference** = the *expert* fascicle mask (dilated as in training) + the *predicted* aponeuroses, measured with the
  run-1 rule. This copies how the experts measured: their fascicles, extended to the aponeuroses.
- **Candidate** = the *predicted* fascicle probabilities + the *same* predicted aponeuroses, measured with the
  candidate rule.
- **Score** = mean(|FL error| in mm / 10 mm, |PA error| / 4°), the stand-in tolerances of `COMPETITION_INFO.md`. The
  FL term uses the 293 images with a high-confidence on-screen ruler. Values are clipped to 30–200 mm as in the
  submission. A rule that finds no fascicle gets that rule's median (as on the test set).
- New GPU work: one 44-second job that ran the v2 aponeurosis ensemble on the fascicle validation images
  (`outputs/pred_masks/val2_fasc_apo/`). The bands look right on these images ([check](figures/fl_apo_check.png)).
- Coverage: 420 images; **11 skipped** because the predicted aponeuroses failed (2.6 %); **57 skipped** because the
  expert mask gives no fascicle line between the predicted bands; **352 scored**.
- **Limits.** The proxy only sees errors from *fascicle prediction and selection*. Aponeurosis errors and scale
  errors are invisible to it, because both sides use the same bands and scale. It also assumes the experts' test
  fascicles behave like the training-mask fascicles. The images come from the training distribution, so it cannot
  see the new scanner, new muscle or patients. A real gain on the leaderboard will be smaller than on the proxy.

### 10.2 Results
One family at a time (all other settings at run-1 values); [figure](figures/fl_tuning.png),
full table `outputs/fl_tuning_families.csv`:

| family | options (run 1 first) | effect on the proxy |
|---|---|---|
| A. number of fascicles | 7 / 3 / 5 / 10 | 3 fascicles (as the experts drew): **+14 % worse**. 5: +0.5 %. 10: −0.9 % |
| B. direction filter | 8° / 5° / 12° | all within ±0.4 % |
| C. min. segment length | 4 % / 8 % / 12 % of height | +11 % / +1 % worse |
| D. aggregation | median / length-weighted / one line at the median angle | +4 % / **+15 % worse** |
| E. confidence filter | none / 0.5 / 0.6 | 0 % / +0.5 % (0.5 only affects the 0.3 retry) |
| F. position | all / middle 80 % | +11 % worse |
| G. shrinkage to MT/sin(PA) | none / w 0.5 / w 0.75 | **+61 % / +18 % worse** |

Top 10 of all combinations (`outputs/fl_tuning.csv`, 3,879 rows; "changes" counts PA-from-run-1 as one change):

| # | rule | changes | score | vs run 1 | FL err mm mean / median | PA err ° mean / median |
|---|---|---|---|---|---|---|
| 1 | `k10_dir12_len0.04_median_pnone_midnone_wnone_paRun1` | 3 | 0.3214 | −2.3 % | 3.89 / 2.50 | 1.015 / 0.742 |
| 2 | `k10_dir12_len0.04_median_p0.5_midnone_wnone_paRun1` | 4 | 0.3214 | −2.3 % | 3.89 / 2.50 | 1.015 / 0.742 |
| 3 | `k10_dir12_len0.04_median_p0.5_midnone_wnone` | 3 | 0.3218 | −2.2 % | 3.89 / 2.50 | 1.017 / 0.739 |
| 4 | `k10_dir12_len0.04_median_pnone_midnone_wnone` | 2 | 0.3218 | −2.2 % | 3.89 / 2.50 | 1.017 / 0.739 |
| 5 | `k10_dir12_len0.04_median_p0.6_midnone_wnone_paRun1` | 4 | 0.3221 | −2.1 % | 3.91 / 2.50 | 1.015 / 0.742 |
| 6 | `k10_dir12_len0.04_median_p0.6_midnone_wnone` | 3 | 0.3235 | −1.7 % | 3.91 / 2.50 | 1.025 / 0.739 |
| 7 | `k10_dir8_len0.04_median_p0.5_midnone_wnone_paRun1` | 3 | 0.3245 | −1.4 % | 3.95 / 2.50 | 1.015 / 0.742 |
| 8 | `k10_dir8_len0.04_median_pnone_midnone_wnone_paRun1` | 2 | 0.3245 | −1.4 % | 3.95 / 2.50 | 1.015 / 0.742 |
| 9 | `k10_dir8_len0.04_median_p0.6_midnone_wnone_paRun1` | 3 | 0.3252 | −1.2 % | 3.97 / 2.50 | 1.015 / 0.742 |
| 10 | `k10_dir12_len0.04_wmedian_pnone_midnone_wnone_paRun1` | 4 | 0.3252 | −1.2 % | 3.97 / 2.66 | 1.015 / 0.742 |
| run 1 | `k7_dir8_len0.04_median_pnone_midnone_wnone` | 0 | 0.3290 | 0 | 4.04 / 2.74 | 1.015 / 0.742 |

**What this says, in plain words.** The run-1 rule is already near the best this family of rules can do. FL's
proxy error is ~4 mm mean / 2.7 mm median, well under the 10 mm tolerance. Ideas that *sound* more stable, such as
deriving FL from one averaged angle or pulling FL towards MT/sin(PA), are clearly worse: real fascicles and
aponeuroses are not perfectly straight and parallel, and per-fascicle measurement captures that. Copying the
experts' "3 fascicles" also hurts, because our 3 longest predicted lines are noisier than the median of 7.

### 10.3 Choice
- Best rule with ≤ 3 changes: `k10 + dir_tol 12° + run-1 PA` (−2.3 %). Runner-up among simpler rules: 10 fascicles
  alone (−0.9 %).
- Plan rule: take the simplest rule within 3 % of the best → that is **run 1 itself** (0 changes). The gain is
  < 3 % and the clip-level bootstrap of the difference (143 clips) has a 95 % interval of [−0.022, +0.022] around
  −0.005. **Conclusion: run 2 found no FL improvement.**
- The best-but-not-significant rule is kept as `measure.py --rule k10dir12` (FL only, PA and MT from run 1) for the
  optional v4/v5 files. Default `--rule run1` regenerates v3 **bit-identically** (checked in job 1607029).

### 10.4 Files written (all comma-separated, 309 rows, `check_submission` passed)
| file | content | MT check |
|---|---|---|
| `submission_v6_avg_v2_v3.csv` | per image: PA and FL = mean of v2 and v3; **MT = v3** | identical to v3 ✓ |
| `submission_v4_k10dir12_v3models.csv` | v3 masks, rule `k10dir12`: only FL changes (PA = v3) | identical to v3 ✓ |
| `submission_v5_k10dir12_v2models.csv` | v2 masks, rule `k10dir12`: only FL changes (PA = v2) | identical to v2 ✓ |

v6 deviates from the plan's literal "average of v2 and v3" in one point: the run-2 hard rule says MT must equal v3
exactly, so MT is not averaged (`DECISIONS.md`). `submission.csv` was not changed.

### 10.5 Test-set checks (reported, not tuned on; `outputs/run2_checks.json`)
| | v2 | v3 | **v4** | v5 | **v6** |
|---|---|---|---|---|---|
| FL median / SD (mm) | 88.4 / 27.5 | 87.0 / 27.5 | 88.2 / 27.1 | 88.9 / 27.2 | 87.1 / 27.3 |
| FL clipped to 30 / 200 mm | 3 / 0 | 3 / 0 | 2 / 0 | 2 / 0 | 3 / 0 |
| within-clip SD, FL mean / median (mm) | 3.33 / 1.55 | 3.85 / 2.71 | **2.92 / 1.44** | 2.93 / 1.67 | 3.29 / 2.40 |
| within-clip SD, PA mean (°) | 0.63 | 0.54 | 0.54 | 0.63 | 0.54 |
| FL·sin(PA)/MT, median (10–90 %) | 1.01 (0.90–1.18) | 1.01 (0.89–1.20) | 1.03 (0.89–1.24) | 1.02 (0.90–1.23) | 1.01 (0.90–1.19) |

- **v4 vs v3:** PA and MT identical. FL changed on 305 images, mean |ΔFL| 2.9 mm, max 16.5 mm, > 5 mm on 69 images.
  FL is **more stable within clips** (SD 3.85 → 2.92 mm), as hoped.
- **v6 vs v3:** mean |ΔPA| 0.29° (max 6.1°), mean |ΔFL| 1.7 mm (max 17.7 mm).
- Largest FL changes v3 → v4 (all from a few clips): IMG_00103 140→123, IMG_00102 140→124, IMG_00146 108→93,
  IMG_00170 90→104, IMG_00101 138→125, IMG_00105 138→125, IMG_00173 98→110, IMG_00058 172→161, IMG_00059 172→161,
  IMG_00172 98→108 mm. [Overlay of 4 clips](figures/fl_changes_v4.png): on IMG_00103/146 the extra fascicles agree
  with the visible fibres. On **IMG_00170/173 the looser 12° filter lets in flatter lines**, which lengthen FL. That
  is the risk of v4. [FL distribution and ΔFL](figures/fl_test_hist_v4.png).
- Plausibility: FL·sin(PA)/MT ≈ 1.01–1.03 for every version, so the three numbers stay geometrically consistent.

### 10.6 Upload order (nothing uploaded; the user uploads)
1. **`submission_v6_avg_v2_v3.csv` first.** It needs no tuning (plan §5). Averaging two independently trained
   model sets usually reduces PA/FL noise, and its FL is more stable within clips than v3.
2. Optional, only if you want to test the FL rule: `submission_v4_k10dir12_v3models.csv`. The difference from v3
   (0.50942) will probably be small, in either direction (the proxy gain was 2.3 %, within noise). Upload `submission_v5_k10dir12_v2models.csv` only if v4 differs from v3 by
   more than 0.03 (surprising in either direction), to confirm the rule's effect on the other model set.
3. Final picks: v3 plus whichever of v6 / v4 scores lower. If neither beats 0.50942, keep v2 + v3.

### 10.7 Compute and rules
- GPU: 1 job, 44 s on an A30 (budget 2 GPU-hours). CPU: 2 `debugq` jobs (2 min 15 s with 16 CPUs; 1 min 21 s with
  4 CPUs). At most 1 Slurm job at a time, no `srun`. Light checks (4 images, CSV files, figures) ran on the login
  node. Wall time ~1 h.
- No retraining, no upload, `data/` and `outputs/cache/` untouched, no existing submission overwritten.
  Report sections 1–9 unchanged. The old "10. Resources" heading was renumbered to 11.
- Rule tuning used only v2 validation predictions. No test values were used for any choice. No filenames or order
  are used at prediction time.
- How to re-run run 2:
  ```bash
  sbatch slurm/predict_fasc_apo.slurm                         # GPU, apo predictions on fasc val images
  sbatch slurm/fl_tune.slurm                                  # CPU, outputs/fl_tuning.csv + v3 regeneration check
  python src/fl_choose.py                                     # rule choice -> outputs/fl_choice.json
  sbatch --export=ALL,RULE=k10dir12 slurm/run2_submit.slurm   # CPU, v4 + v5
  python src/run2_checks.py --v4 submission_v4_k10dir12_v3models.csv --v5 submission_v5_k10dir12_v2models.csv  # v6 + checks
  ```

## 11. Resources and rule notes (run 1)
- ~4.5 GPU-hours (A30) over 39 Slurm jobs, ~0.5 CPU-job hours, about 4 h wall time. Well within budget.
- Nothing was uploaded to Kaggle, and nothing in `data/` was changed.
- A few short CPU checks (`srun`, seconds to minutes) ran while two GPU jobs were active, briefly exceeding the
  "max 2 jobs at once" rule. One extra GPU job was queued by mistake and cancelled before it started.
- External code: DL_Track_US 0.3.0 (Apache-2.0) was **read** for its measurement definitions (inner-edge thickness,
  uppermost band pair) and ruler calibration idea. No code was copied. Other licences: see `README.md`.
- Environment: `outputs/environment.txt` (pip freeze), `requirements.txt`.

## Figures
[crop_check](figures/crop_check.png) · [preprocess_examples](figures/preprocess_examples.png) ·
[neighbour_frames_apo](figures/neighbour_frames_apo.png) · [neighbour_frames_fasc](figures/neighbour_frames_fasc.png) ·
[train_apo_A](figures/train_apo_A.png) · [train_apo_B](figures/train_apo_B.png) · [train_fasc_A](figures/train_fasc_A.png) ·
[train_fasc_B](figures/train_fasc_B.png) · experiments [A](figures/train_exp_fasc_A.png) [B](figures/train_exp_fasc_B.png)
[C](figures/train_exp_fasc_C.png) [D](figures/train_exp_fasc_D.png) ·
[val_examples_apo_final](figures/val_examples_apo_final.png) · [val_examples_fasc_final](figures/val_examples_fasc_final.png) ·
[test_hist (v2)](figures/test_hist_submission_v2_apoAB_fascA.png) ·
[test_measurements (v2)](figures/test_measurements_submission_v2_apoAB_fascA.png) ·
v1: [hist](figures/test_hist_submission_v1_baseline.png), [measurements](figures/test_measurements_submission_v1_baseline.png) ·
run 2: [fl_tuning](figures/fl_tuning.png) · [fl_apo_check](figures/fl_apo_check.png) · [fl_test_hist_v4](figures/fl_test_hist_v4.png) ·
[fl_changes_v4](figures/fl_changes_v4.png) · [test_measurements (v4)](figures/test_measurements_submission_v4_k10dir12_v3models.png)
