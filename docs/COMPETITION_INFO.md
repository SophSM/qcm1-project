# Competition info: UMUD Challenge (Muscle Architecture in Ultrasound Data)

Sources: the Kaggle **Overview** and **Data** pages and the official **rules** (all pasted by the user on
2026-11-14), plus the Kaggle API. This is a condensed version: everything relevant for building a solution is kept.

## Key facts
| | |
|---|---|
| URL | https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data |
| Task | Estimate **pennation angle (PA, °)**, **fascicle length (FL, mm)**, **muscle thickness (MT, mm)** for each of 309 test images |
| Deadline | 2026-11-14 23:00 UTC. **The user may submit late** (scored, but not ranked), so this is not a constraint. |
| Submissions | max 5 per day. Up to 2 can be chosen as final. |
| Leaderboards | Public LB = part of the test set. Final ranking = private LB. |
| Prizes | CHF 1500 / 1000 / 500. Team size ≤ 10. |
| Data licence | CC BY-NC-SA 4.0: non-commercial / academic use only |
| Sponsor | Swissuniversities / University of Basel |
| Methods | Classic computer vision, deep learning or hybrid: all welcome. The training masks *may* be used. |

## Evaluation: `UMUD_score`
- **Normalised mean absolute error** over PA, FL and MT. Each variable's error is divided by a **predefined
  tolerance**, so the three count comparably despite different units. **Lower is better.**
- The tolerance values are **not published**. A reasonable stand-in: the raters' discussion thresholds **PA 4°,
  FL 10 mm, MT 1 mm**. If the real tolerances are similar, **MT errors of ~1 mm already cost as much as 4° of PA**.

## How the reference (ground-truth) values were made
- Two researchers annotated each test image manually and their results were **averaged**. Disagreements of
  > 10 mm FL, > 4° PA or > 1 mm MT were discussed and re-evaluated.
- Per image they marked: superficial aponeurosis, deep aponeurosis, **3 visible fascicles**, **3 pennation
  angles**, **3 muscle-thickness measurements**.
- **PA** = angle between the fascicle line and the **deep** aponeurosis.
- **FL** = distance between the superficial and deep aponeuroses **along the fascicle line**. When the fascicle
  leaves the image, it is extended with **linear extrapolation**.
- **MT** = **perpendicular** distance between the aponeuroses, at **3 locations spread over the muscle width**.
- **Test-set ranges:** PA **5–45°**, FL **30–200 mm**, MT **10–50 mm**.

## Data
**Images:** 2-D B-mode ultrasound of human leg muscles. Fascicles appear as slanted bright lines. The aponeuroses
appear as brighter, roughly parallel bands. **No pre-processing** was applied apart from removing participant
identifiers, so on-screen elements such as scales should still be there.

| | Training | Test |
|---|---|---|
| Size | 2,761 fascicle images + masks. 1,048 aponeurosis images + masks (unaugmented). | 309 images (251 `.tif`, 58 `.png`), "with released architecture measurements" (= hidden labels) |
| Source | Published data, Ritsche et al. (2024) | **Unpublished data**, participants **not in training** |
| Scanners | Siemens Acuson Juniper, Telemed ArtUS EXT-1H, Telemed Echo Blaster 128, Hitachi Aloka Alpha-10, Philips HD11 | Siemens Acuson Juniper, Telemed ArtUS EXT-1H, **Philips Lumify (new)** |
| Muscles | vastus lateralis, gastrocnemius medialis & lateralis, tibialis anterior | vastus lateralis, gastrocnemius medialis, **rectus femoris (new)** |
| Participants | healthy, mostly 18–35 y (63% male), ~10% aged 60–70 | healthy **and cerebral palsy**, 18–28 y (59% male) |
| Regions | distal, middle, proximal | distal, middle, proximal |

- The test set includes **sequences of 5 consecutive video frames**, used to judge the **temporal stability** of
  algorithms.
- ⚠️ The training set contains **exact duplicate images** (listed in a Kaggle discussion). They don't affect the
  leaderboard, but they **inflate internal validation scores** unless duplicates are kept on the same side of the
  split.
- **No pixel spacing / mm scale is provided.** It must be derived from the images (e.g. on-screen scale bars).
- Files: `apo_imgs`, `apo_masks`, `fasc_imgs`, `fasc_masks`, `test_images`, `sample_submission.csv`
  (`image_id;pa_deg;fl_mm;mt_mm`, `;`-separated, `image_id` with file extension).

## Reference methods named by the organisers
- **DL_Track_US** (Ritsche et al., the organisers' own tool). The organisers uploaded the **DL_Track_US 0.3.1
  notebook for aponeurosis prediction** on Kaggle; earlier versions also predicted fascicles. Training data follows
  its approach ("UMB", Ritsche et al. 2024).
- **UltraTimTrack**, **DUSTrack**.
- "Submissions must be based on **novel algorithm/model development**. Existing code can be re-used in an improved
  version."

## Rules that matter for how we build the solution
1. **Fully automated predictions only.** No manual annotation or correction of test images.
2. **No per-image tuning from leaderboard feedback.** No values hard-coded to filenames, clip IDs or input order.
   Global choices (model, ensemble, hyperparameters) are fine.
3. **Must run on new data:** the organisers may re-run the code on test images with **anonymised filenames in random
   order**, and on **new images**. Ensembles are judged by "can it be re-run on a new dataset where no leaderboard
   exists?". Prediction must never depend on filenames or order.
4. **No test labels from other sources**, e.g. public datasets containing the same images.
5. **Allowed external resources:** the provided training data, public datasets **indexed in UMUD**, public pretrained
   models. **Must be documented** in the method description.
6. **Open-source licences only** (OSI-approved, allowing commercial use). Our stack is OK: PyTorch (BSD),
   segmentation-models-pytorch (MIT), albumentations (MIT), timm (Apache-2.0), scikit-image (BSD), OpenCV
   (Apache-2.0). Check the licence of any reused external code (e.g. DL_Track_US) before copying.
7. **Data security:** don't share or publish the competition data.
8. No private code sharing outside a team. Public sharing only on Kaggle forums/notebooks.

## Prize eligibility (only relevant if placed top 3)
After the deadline, the provisional top 3 must provide, **within 14 days**:
- a **public repository** under an **OSI licence** (code without a licence is rejected; the winner licence is
  GPL-3.0);
- that meets **FAIR software** standards (findable, accessible, documented, with metadata);
- that is **reproducible**: notebook, script, package or installer, plus `requirements.txt` / `environment.yml` /
  Docker, and instructions with exact commands to reproduce inference;
- plus a method description: architecture, preprocessing, loss, training details, hyperparameters, external
  data/models.

The organisers verify by re-running the code, including on renamed/shuffled test images and new images. Results
must match within tolerance. Decisions are announced ~3 weeks after the deadline.
