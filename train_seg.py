import os
import cv2
import torch
import random
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
import torch.nn.functional as F
from tqdm import tqdm

from src.utils import degrade_image

DEVICE = "cuda"
IMG_SIZE = 512


# =========================
# Dataset
# =========================
class VesselDataset(Dataset):
    def __init__(self, img_dir, mask_dir, dataset_type, use_degrade=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type
        self.use_degrade = use_degrade

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".tif", ".jpg", ".png", ".ppm"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        # ===== image =====
        img = cv2.imread(os.path.join(self.img_dir, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # ===== mask mapping =====
        if self.dataset_type == "drive":
            num = name.split("_")[0]
            mask_name = f"{num}_manual1.gif"

        elif self.dataset_type == "hrf":
            mask_name = name.replace(".jpg", ".tif")

        elif self.dataset_type == "chase":
            mask_name = name.replace(".jpg", "_1stHO.png")

        else:
            raise ValueError("Unknown dataset")

        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), 0)

        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)

        # ===== augment =====
        if random.random() < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        if random.random() < 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)

        # ===== domain gap =====
        if self.use_degrade and random.random() < 0.5:
            img = degrade_image(img, severity=2)

        # ===== normalize =====
        img = img.astype(np.float32) / 255
        img = (img - 0.5) / 0.5

        img = torch.from_numpy(img.copy()).permute(2, 0, 1)
        mask = torch.from_numpy(mask.copy()).unsqueeze(0)

        return img, mask


# =========================
# Loss
# =========================
def bce_dice_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask)

    pred = torch.sigmoid(pred)

    inter = (pred * mask).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3))

    dice = (2 * inter + 1e-8) / (union + 1e-8)
    dice_loss = 1 - dice.mean()

    return bce + dice_loss

def build_dataset(dataset):
    if dataset == "drive":
        return VesselDataset(
            "./data/test/drive_images",
            "./data/test/drive_masks",
            "drive"
        )

    elif dataset == "hrf":
        return VesselDataset(
            "./data/test/hrf_images",
            "./data/test/hrf_masks",
            "hrf"
        )

    elif dataset == "chase":
        return VesselDataset(
            "./data/test/chase_images",
            "./data/test/chase_masks",
            "chase"
        )

    elif dataset == "mixed":
        d1 = VesselDataset("./data/test/drive_images", "./data/test/drive_masks", "drive")
        d2 = VesselDataset("./data/test/hrf_images", "./data/test/hrf_masks", "hrf")
        d3 = VesselDataset("./data/test/chase_images", "./data/test/chase_masks", "chase")

        return torch.utils.data.ConcatDataset([d1, d2, d3])

    else:
        raise ValueError("Invalid dataset")

# =========================
# Train
# =========================
def train(dataset):
    dataset = build_dataset(dataset)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"Training on {dataset} dataset...")

    for epoch in range(100):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

        for img, mask in pbar:
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            pred = model(img)
            loss = bce_dice_loss(pred, mask)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    os.makedirs("./segmentation", exist_ok=True)
    torch.save(model.state_dict(), f"./segmentation/{dataset}_seg_unet.pth")

    print(f"Model saved: {dataset}_seg_unet.pth")


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="drive",
                        choices=["drive", "hrf", "chase", "mixed"])

    args = parser.parse_args()

    train(args.dataset)