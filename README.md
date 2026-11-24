# QCM Kaggle project: muscle architecture from ultrasound images

Our entry in the Kaggle **UMUD Challenge (Muscle Architecture in Ultrasound Data)**. For each B-mode ultrasound
image of a leg muscle, the pipeline estimates three values fully automatically:

| Value | Unit | Definition |
|---|---|---|
| Pennation angle (PA) | ° | angle between the fascicles and the deep aponeurosis |
| Fascicle length (FL) | mm | length of a fascicle between the two aponeuroses, extrapolated in a straight line |
| Muscle thickness (MT) | mm | perpendicular distance between the aponeuroses, at 3 places across the width |

**Best leaderboard score: 0.50942** (submission v3; lower is better).

## How it works

```
screenshot ──► 1. crop the scan ──► 2. prepare inputs ──► 3. two U-Nets ──► 4. geometry ──► 5. pixel → mm ──► 6. checks ──► PA, FL, MT
```

1. **Crop** (`src/crop.py`): find the grey, textured scan area and drop the screen text, rulers and colour bars.
   The crop depends only on image content, never on file names.
2. **Prepare** (`src/preprocess.py`): the aponeurosis model sees the whole crop resized to 512 × 512. The fascicle
   model sees the crop at 768 px height and works on 512 × 512 tiles. Fascicle masks are thickened to about 5 px.
3. **Segment** (`src/model.py`, `src/train.py`): two U-Nets (segmentation-models-pytorch, ResNet34 encoder with
   ImageNet weights). One draws the aponeuroses, the other draws the fascicles.
4. **Measure** (`src/geometry.py`, `src/measure.py`): fit lines to the predicted bands and fibres and measure
   MT, PA and FL the way the expert protocol describes.
5. **Scale** (`src/scale.py`): read the on-screen ruler ticks to convert pixels to millimetres.
6. **Check**: measurements that fail take the median of the other images, and values outside the official
   ranges are clipped.

### Submission v3 in detail

| Model | Input | Training images | Epochs |
|---|---|---|---|
| Aponeurosis U-Net A | grey image in all 3 channels | all 1,048 | 46 |
| Aponeurosis U-Net B | grey + CLAHE (local contrast) + Sato ridge filter | all 1,048 | 46 |
| Fascicle U-Net | grey image in all 3 channels | all 2,761 | 19 |

- **Ensemble:** the aponeurosis probability maps of A and B are averaged.
- **Test-time augmentation:** every image is also predicted mirrored left–right, and the two results are averaged.
- **Training:** loss 0.5 · Dice + 0.5 · BCE; AdamW with lr 1e-4 and a cosine schedule; batch 8; mixed precision;
  seed 42.
- **Augmentation:** horizontal flip, affine (±10 % scale/shift, ±10°), brightness/contrast ±20 % and gamma 80–120 %.
  The aponeurosis model also gets a "narrow window" augmentation that imitates the narrow scans of the new test
  scanner.
- **Epoch counts:** v2 trained on 85 % of the images and used the other 15 % to choose them. v3 retrains on 100 %
  of the images with the same epoch counts.

### Validation (v2 models, held-out 15 %)

| | Aponeurosis | Fascicle |
|---|---|---|
| Dice vs expert masks | 0.83 | 0.58 |
| Measurement error | MT 0.55 mm mean (0.18 mm median) | dominant angle 0.75° |

The validation split keeps exact duplicates and video clips on one side (`src/split.py`), so near-copies never
inflate the score. Input enhancement was tested as an experiment (grey vs CLAHE + ridge filter vs grain removal):
plain grey was best for fascicles. Full numbers and discussion are in [`docs/REPORT.md`](docs/REPORT.md).

| Submission | What changed | Leaderboard |
|---|---|---|
| v2 | aponeurosis A+B ensemble, fascicle A, trained on 85 % | 0.51394 |
| **v3** | same recipe, trained on all labelled images | **0.50942** |
| v4–v6 | different fascicle-length rules (run 2) | no reliable gain on validation (see report §10) |

## Repository layout

```
├── src/                     pipeline code
│   ├── config.py            paths and training defaults
│   ├── data.py              image and mask loading
│   ├── crop.py              content-based scan crop
│   ├── split.py             duplicate- and clip-aware train/validation split
│   ├── preprocess.py        crops, model inputs, input cache
│   ├── model.py             U-Net, loss, sliding-window inference
│   ├── train.py             training (apo / fasc, variants A–D, --all_data)
│   ├── predict.py           full inference chain → submission CSV
│   ├── evaluate.py          Dice/IoU, thickness and angle errors on validation
│   ├── geometry.py          bands, fascicle lines, PA / FL / MT in pixels
│   ├── scale.py             ruler-tick reading: pixels → mm
│   ├── measure.py           measurement rules and per-image measurements
│   └── fl_tune.py, fl_choose.py, run2_checks.py   run 2: fascicle-length rule tuning
├── slurm/                   job scripts for the Slurm cluster (prep, train, predict, …)
├── notebooks/
│   └── data_exploration.ipynb   first look at the data (outputs stripped)
├── slides/                  scripts that build the presentation decks
├── docs/
│   ├── REPORT.md            full technical report (results, decisions, commands)
│   ├── DECISIONS.md         short log of design decisions
│   ├── COMPETITION_INFO.md  task, metric, data and rules in brief
│   ├── environment.txt      exact package versions of the run
│   ├── figures/             training curves, tuning and prediction histograms
│   └── slides/              presentation decks (full and 10-minute version)
└── requirements.txt
```

## Setup

```bash
conda create -p <env> python=3.11
conda activate <env>
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The Slurm scripts activate the environment given in `CONDA_ENV`: `export CONDA_ENV=<env>` before calling
`sbatch`. Exact versions of the original run are in `docs/environment.txt`. Hardware used: one NVIDIA A30 (24 GB)
per training job.

## Data

The competition data is **not included**. Its licence (CC BY-NC-SA 4.0) and the competition rules do not allow
sharing it. Download it from the
[Kaggle competition page](https://www.kaggle.com/competitions/umud-challenge-muscle-architecture-in-ultrasound-data)
and put it under `data/` (or symlink it). `src/config.py` expects the folders `apo_imgs_v1`, `apo_masks_v1`,
`fasc_imgs_v1`, `fasc_masks_v1` and `test_images_v2`, plus `sample_submission.csv`.

Trained weights (`models/`) and run outputs (`outputs/`) are not included either.

## Reproduce

From the repository root, on a Slurm cluster:

```bash
# 1. crops, split and input cache (CPU, ~25 min)
sbatch slurm/prep.slurm

# 2. v2 models (85 % training, 15 % validation)
sbatch slurm/train.slurm --task apo  --variant A --epochs 60 --patience 15 --name apo_A
sbatch slurm/train.slurm --task apo  --variant B --epochs 60 --patience 15 --name apo_B
sbatch slurm/train.slurm --task fasc --variant A --epochs 60 --patience 12 --name fasc_A

# 3. v3 models (all labelled images, epoch counts chosen by v2)
sbatch slurm/train.slurm --task apo  --variant A --epochs 46 --all_data --save_last --name apo_A_all
sbatch slurm/train.slurm --task apo  --variant B --epochs 46 --all_data --save_last --name apo_B_all
sbatch slurm/train.slurm --task fasc --variant A --epochs 19 --all_data --save_last --name fasc_A_all

# 4. v3 submission
sbatch --export=ALL,APO="models/apo_A_all.pt models/apo_B_all.pt",FASC=models/fasc_A_all.pt,TAG=v3,OUT=submission_v3_alldata.csv slurm/predict.slurm
```

Variant A is the grey image only. Variant B is grey + CLAHE + ridge filter. Without Slurm, run the `python src/...`
commands inside the job scripts directly. The run-2 commands (v4–v6) are in `docs/REPORT.md` §10.

Predictions never use file names or file order: crop, scale and measurements come from each image's own content.
This is required by the competition, where the organisers may re-run the code on renamed, shuffled images.

## Slides

`docs/slides/` holds the presentation of submission v3: a full deck and a 10-minute version. In the short version
the background slides are hidden but still in the file. To rebuild them:

```bash
pip install python-pptx
# download the Slidesgo "Elegant Black and White Thesis Defense" template to slides/style_template.pptx
python slides/build_deck.py docs/slides/submission_v3_presentation.pptx
python slides/build_deck.py docs/slides/submission_v3_presentation_short.pptx short
```

The decks use the fonts Figtree and Hanken Grotesk (free on Google Fonts). All diagrams are drawn; no competition
images are used. Template by Slidesgo.

## External code, models and data

| Item | Use | Licence |
|---|---|---|
| ImageNet-pretrained ResNet34 (via segmentation-models-pytorch) | encoder initialisation | MIT (smp), BSD (torchvision weights) |
| PyTorch, torchvision | training and inference | BSD-3 |
| segmentation-models-pytorch | U-Net | MIT |
| timm | encoder backbone code | Apache-2.0 |
| albumentations | augmentation | MIT |
| OpenCV | image operations | Apache-2.0 |
| scikit-image, SciPy, NumPy, pandas, Pillow, matplotlib | filters, geometry, I/O, figures | BSD / HPND / PSF-compatible |
| DL_Track_US 0.3.0 (Ritsche et al.) | read only, for the measurement definitions and the idea of ruler-based calibration; no code copied | Apache-2.0 |

No external data was used: only the provided training images and masks.
