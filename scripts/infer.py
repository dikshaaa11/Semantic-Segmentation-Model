import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from tqdm import tqdm

from models.unet import UNet

DEVICE = "cpu"
IMG_SIZE = 64
MODEL_PATH = "outputs/best_model.pth"
TEST_DIR = "data/test"
SAVE_DIR = "outputs/predictions"

os.makedirs(SAVE_DIR, exist_ok=True)

# Transform (same as training)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load model
model = UNet(in_channels=3, num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")

# Get all test images
image_files = os.listdir(TEST_DIR)

for img_name in tqdm(image_files):
    img_path = os.path.join(TEST_DIR, img_name)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Save raw mask
    save_path = os.path.join(SAVE_DIR, img_name)
    cv2.imwrite(save_path, pred.astype(np.uint8))

print("Inference complete.")