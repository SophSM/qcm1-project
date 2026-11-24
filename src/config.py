"""Central settings: paths and training defaults. Change things here, not inside the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"            # symlink to the kagglehub cache
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

DIRS = {
    "apo":  {"imgs": DATA / "apo_imgs_v1/apo_images_new_model_v1",
             "masks": DATA / "apo_masks_v1/apo_masks_new_model_v1"},
    "fasc": {"imgs": DATA / "fasc_imgs_v1/fasc_images_new_model_v1",
             "masks": DATA / "fasc_masks_v1/fasc_masks_new_model_v1"},
}
TEST_DIR = DATA / "test_images_v2/test_set_v2"
SAMPLE_SUBMISSION = DATA / "sample_submission.csv"   # ';'-separated

# Training defaults (starting points, to be tuned)
IMG_SIZE = 512          # don't go much lower: fascicle lines are very thin
ENCODER = "resnet34"    # pretrained on ImageNet
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
VAL_FRACTION = 0.15     # split in blocks of consecutive files (neighbouring frames may be near-duplicates)
SEED = 42
