# depth_estimation_batch.py
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import numpy as np
import os
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from zoedepth.utils.misc import colorize

# Setup
img_dir = "./data/bicycle/images"  # path to your training images
output_dir = "./data/bicycle/depths"  # where to save the .npy depth maps
os.makedirs(output_dir, exist_ok=True)

# Load ZoeDepth model
config = get_config("zoedepth", "infer")
model = build_model(config)
model.eval().cuda()

# Transform
transform = Compose([Resize((512, 512)), ToTensor()])

# Process all images
for fname in os.listdir(img_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(img_dir, fname)
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).cuda()

    with torch.no_grad():
        pred = model.infer(img_tensor)[0][0]  # shape [H, W]

    # Save .npy depth map
    out_name = os.path.splitext(fname)[0] + ".npy"
    np.save(os.path.join(output_dir, out_name), pred.cpu().numpy())

    print(f"Saved: {out_name}")
