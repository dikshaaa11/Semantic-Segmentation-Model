# Desert Semantic Segmentation

Backend model for desert terrain semantic segmentation using U-Net.

## Structure

- models/ → Model architecture
- scripts/ → Training and inference scripts
- Hackethon/ → Hackathon utilities
- requirements.txt → Dependencies

## How to Run

### Install dependencies
pip install -r requirements.txt

### Train
python -m scripts.train

### Inference
python -m scripts.infer

---

Outputs are saved inside the outputs/ directory (ignored from git).
