"""
Segmentation Inference Script (PURE INFERENCE MODE)

- Loads trained segmentation head
- Runs inference on test images (NO MASKS REQUIRED)
- Saves:
    * Raw predicted masks
    * Colored predicted masks
    * Image + prediction comparison visualizations
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Color Palette & Classes
# ============================================================================

value_map = {
    0: 0,
    100: 1,
    200: 2,
    300: 3,
    500: 4,
    550: 5,
    700: 6,
    800: 7,
    7100: 8,
    10000: 9
}

n_classes = len(value_map)

color_palette = np.array([
    [0,   0,   0  ],   # class 0 - background
    [34,  139, 34 ],   # class 1
    [0,   255, 0  ],   # class 2
    [210, 180, 140],   # class 3
    [139, 90,  43 ],   # class 4
    [128, 128, 0  ],   # class 5
    [139, 69,  19 ],   # class 6
    [128, 128, 128],   # class 7
    [160, 82,  45 ],   # class 8
    [135, 206, 235],   # class 9
], dtype=np.uint8)


def mask_to_color(mask):
    """Convert a class-index mask to an RGB color image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset (IMAGES ONLY - no masks required)
# ============================================================================

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = data_dir
        self.transform = transform
        self.data_ids = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, data_id


# ============================================================================
# Model Definition (must match training architecture)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Visualization
# ============================================================================

def save_prediction_comparison(img_tensor, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of the input image and predicted mask."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(pred_color)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation Inference Script')
    parser.add_argument('--model_path',  type=str, default=os.path.join(script_dir, 'segmentation_head.pth'),
                        help='Path to trained segmentation head weights (.pth)')
    parser.add_argument('--data_dir',    type=str, default='data/test',
                        help='Directory containing test images (no masks needed)')
    parser.add_argument('--output_dir',  type=str, default='./predictions',
                        help='Root directory for all saved outputs')
    parser.add_argument('--batch_size',  type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of comparison visualizations to save')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Image dimensions snapped to multiples of 14 (DINOv2 patch size)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])

    dataset = TestDataset(data_dir=args.data_dir, transform=transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Found {len(dataset)} test images in '{args.data_dir}'")

    # ------------------------------------------------------------------
    # Load DINOv2 backbone (frozen)
    # ------------------------------------------------------------------
    BACKBONE_SIZE = "small"
    backbone_archs = {
        "small": "vits14",
        "base":  "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }

    backbone_name  = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"
    backbone_model = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone_model.eval()
    backbone_model.to(device)

    # Probe embedding dimension from a single forward pass
    sample_img, _ = dataset[0]
    with torch.no_grad():
        probe = backbone_model.forward_features(
            sample_img.unsqueeze(0).to(device)
        )["x_norm_patchtokens"]
    n_embedding = probe.shape[2]
    print(f"DINOv2 embedding dim: {n_embedding}")

    # ------------------------------------------------------------------
    # Load segmentation head
    # ------------------------------------------------------------------
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier.to(device)
    classifier.eval()
    print(f"Loaded segmentation head from '{args.model_path}'")

    # ------------------------------------------------------------------
    # Output sub-directories
    # ------------------------------------------------------------------
    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')

    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    print("Running inference...")
    sample_count = 0

    with torch.no_grad():
        for imgs, data_ids in tqdm(loader, desc="Batches"):
            imgs = imgs.to(device)

            features = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits   = classifier(features)
            outputs  = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            predicted_masks = torch.argmax(outputs, dim=1)  # (B, H, W)

            for i in range(imgs.shape[0]):
                data_id   = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)

                # 1) Raw class-index mask
                Image.fromarray(pred_mask).save(
                    os.path.join(masks_dir, f"{base_name}_pred.png")
                )

                # 2) Colored mask
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f"{base_name}_pred_color.png"),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                )

                # 3) Side-by-side comparison (limited to --num_samples)
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i],
                        predicted_masks[i],
                        os.path.join(comparisons_dir, f"{base_name}_comparison.png"),
                        data_id
                    )

                sample_count += 1

    print(f"\nInference complete. Processed {sample_count} images.")
    print(f"  Raw masks    -> {masks_dir}")
    print(f"  Color masks  -> {masks_color_dir}")
    print(f"  Comparisons  -> {comparisons_dir}")


if __name__ == "__main__":
    main()