import os
import cv2
import torch
import random
import hashlib
import numpy as np
from torch.utils.data import Dataset
import albumentations as A

from .utils import degrade_image


class FundusDataset(Dataset):
    def __init__(self, data_dir, split_file, mode='train', image_size=512):
        self.data_dir = data_dir
        self.mode = mode
        self.image_size = image_size

        with open(split_file, 'r') as f:
            self.image_files = [line.strip() for line in f]

        if self.mode == 'train':
            self.spatial_transform = A.Compose([
                A.SmallestMaxSize(max_size=self.image_size),
                A.CenterCrop(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        else:
            self.spatial_transform = A.Compose([
                A.SmallestMaxSize(max_size=self.image_size),
                A.CenterCrop(self.image_size, self.image_size)
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)

        clean_img = cv2.imread(img_path)
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)

        augmented = self.spatial_transform(image=clean_img)
        clean_img = augmented['image']

        if self.mode == 'train':
            degraded_img = self.apply_degradation(clean_img.copy())
        else:
            seed_val = int(hashlib.md5(img_name.encode()).hexdigest(), 16) % (2**32)
            np.random.seed(seed_val)
            random.seed(seed_val)

            degraded_img = degrade_image(clean_img.copy(), severity=2)

            np.random.seed()
            random.seed()

        clean_tensor = torch.from_numpy(clean_img).permute(2, 0, 1).float() / 255.0
        degraded_tensor = torch.from_numpy(degraded_img).permute(2, 0, 1).float() / 255.0

        return degraded_tensor, clean_tensor

    def apply_degradation(self, img):
        return degrade_image(img, severity=random.uniform(1.0, 3.0))