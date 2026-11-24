"""Loading image/mask pairs. Datasets and augmentations come next (see SESSION_LOG.md, step 2).

Data quirks this module handles:
- some TIFFs are LZW-compressed -> read with Pillow, not tifffile
- some masks are palette ("P") images -> convert to grayscale before thresholding
- 1200x800 images have masks stored at another size -> resize mask to image (nearest-neighbour)
"""
import os

import numpy as np
from PIL import Image

IMG_EXT = (".tif", ".tiff", ".png", ".jpg")


def list_images(d):
    """Sorted image filenames in a folder (skips Thumbs.db etc.)."""
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def load_img(p, gray=True):
    im = Image.open(p)
    return np.array(im.convert("L" if gray else "RGB"))


def load_mask(p, size=None):
    """Binary mask (bool). `size` = (width, height) of the image to align it with."""
    m = Image.open(p).convert("L")
    if size is not None and m.size != tuple(size):
        m = m.resize(tuple(size), Image.NEAREST)
    return np.array(m) > 127


def load_pair(img_path, mask_path, gray=True):
    img = load_img(img_path, gray)
    mask = load_mask(mask_path, size=(img.shape[1], img.shape[0]))
    return img, mask

# TODO: PyTorch Dataset with resize to config.IMG_SIZE + augmentations
#       (left-right flip, small rotation/zoom, brightness/contrast; NO upside-down flips)
# TODO: block-wise train/validation split
