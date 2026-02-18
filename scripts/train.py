import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from models.unet import UNet
from scripts.dataset import DesertSegmentationDataset

DEVICE = "cpu"
BATCH_SIZE = 2
EPOCHS = 4
LR = 1e-4
IMG_SIZE = 64

def compute_iou(preds, masks, num_classes=10):
    preds = torch.argmax(preds, dim=1)
    ious = []

    for cls in range(num_classes):
        intersection = ((preds == cls) & (masks == cls)).sum()
        union = ((preds == cls) | (masks == cls)).sum()

        if union == 0:
            continue

        ious.append((intersection / union).item())

    if len(ious) == 0:
        return 0.0

    return np.mean(ious)


def main():

    train_dataset = DesertSegmentationDataset(
        image_dir="data/train/images",
        mask_dir="data/train/masks",
        img_size=IMG_SIZE
    )

    val_dataset = DesertSegmentationDataset(
        image_dir="data/val/images",
        mask_dir="data/val/masks",
        img_size=IMG_SIZE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = UNet(in_channels=3, num_classes=10).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_iou = 0

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_iou = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(images)
                val_iou += compute_iou(preds, masks)

        val_iou /= len(val_loader)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print("Model Saved!")

    print("\nTraining Complete.")


if __name__ == "__main__":
    main()