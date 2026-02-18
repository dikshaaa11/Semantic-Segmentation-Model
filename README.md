# Desert Semantic Segmentation Backend

This repository contains the backend implementation for desert terrain semantic segmentation.

## Features
- U-Net based segmentation model
- Training pipeline
- Inference script for test images
- DINOv2 backbone support

## Folder Structure
- models/ → U-Net architecture
- scripts/ → training & inference scripts
- Hackathon/ → environment setup files
- outputs/ → model weights & predictions (ignored in git)
- data/ → dataset (ignored in git)

## How to Run

### 1. Create environment
conda create -n segmentation python=3.10
conda activate segmentation
pip install -r requirements.txt

### 2. Train
python -m scripts.train

### 3. Inference
python -m scripts.infer
