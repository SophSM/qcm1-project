"""U-Net (ResNet34 encoder, ImageNet weights), loss, input normalisation and sliding-window inference."""
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

import config

MEAN = np.array([0.45, 0.45, 0.45], np.float32)
STD = np.array([0.225, 0.225, 0.225], np.float32)


def build_model(pretrained=True):
    return smp.Unet(encoder_name=config.ENCODER, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


class DiceBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        return 0.5 * self.dice(logits, target) + 0.5 * self.bce(logits, target)


def to_tensor(img):
    """HxWx3 uint8 -> 3xHxW float tensor (normalised)."""
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


@torch.no_grad()
def predict_whole(model, x, device, tta=True):
    """x: 3xHxW tensor (H, W multiples of 32). Returns HxW probability (numpy)."""
    xb = x[None].to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        p = torch.sigmoid(model(xb).float())
        if tta:
            p = 0.5 * (p + torch.flip(torch.sigmoid(model(torch.flip(xb, [3])).float()), [3]))
    return p[0, 0].cpu().numpy()


@torch.no_grad()
def predict_sliding(model, x, device, patch=512, stride=256, tta=True, batch=8):
    """Sliding-window prediction with 50 % overlap, averaged. x: 3xHxW tensor. Returns HxW probability."""
    _, H, W = x.shape
    ph, pw = max(H, patch), max(W, patch)
    if (ph, pw) != (H, W):
        x = torch.nn.functional.pad(x[None], (0, pw - W, 0, ph - H), mode="reflect")[0]
    ys = list(range(0, ph - patch + 1, stride))
    xs = list(range(0, pw - patch + 1, stride))
    if ys[-1] != ph - patch:
        ys.append(ph - patch)
    if xs[-1] != pw - patch:
        xs.append(pw - patch)
    acc = torch.zeros((ph, pw), device=device)
    cnt = torch.zeros((ph, pw), device=device)
    coords = [(y, xx) for y in ys for xx in xs]
    for i in range(0, len(coords), batch):
        cc = coords[i:i + batch]
        xb = torch.stack([x[:, y:y + patch, xx:xx + patch] for y, xx in cc]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(xb).float())
            if tta:
                p = 0.5 * (p + torch.flip(torch.sigmoid(model(torch.flip(xb, [3])).float()), [3]))
        for (y, xx), pp in zip(cc, p[:, 0]):
            acc[y:y + patch, xx:xx + patch] += pp
            cnt[y:y + patch, xx:xx + patch] += 1
    return (acc / cnt)[:H, :W].cpu().numpy()


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck
