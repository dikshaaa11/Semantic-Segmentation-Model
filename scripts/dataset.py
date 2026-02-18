import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch


# Mapping original dataset IDs to 0–9
CLASS_MAPPING = {
    100: 0,    # Trees
    200: 1,    # Lush Bushes
    300: 2,    # Dry Grass
    500: 3,    # Dry Bushes
    550: 4,    # Ground Clutter
    600: 5,    # Flowers
    700: 6,    # Logs
    800: 7,    # Rocks
    7100: 8,   # Landscape
    10000: 9   # Sky
}


def remap_mask(mask):
    new_mask = np.zeros_like(mask)
    for original_id, new_id in CLASS_MAPPING.items():
        new_mask[mask == original_id] = new_id
    return new_mask


class DesertSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg"))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith((".png", ".jpg"))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask = remap_mask(mask)

        image = image / 255.0
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)