# depth_estimation.py
# This script demonstrates how to use the ZoeDepth model for depth estimation.

import torch
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import numpy as np
import os

from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model
from zoedepth.utils.misc import colorize

# 1. Load ZoeDepth
config = get_config("zoedepth", "infer")
model = build_model(config)
model.eval().cuda()

# 2. Load your style image
img_path = "../gaussian_splatting/style/starrynight.jpg"
image = Image.open(img_path).convert("RGB")
transform = Compose([Resize((512, 512)), ToTensor()])
img_tensor = transform(image).unsqueeze(0).cuda()  # shape: [1, 3, H, W]

# 3. Predict depth
with torch.no_grad():
    pred = model.infer(img_tensor)[0][0]  # shape: [H, W]

# 4. Save raw depth as .npy
np.save("style_depth.npy", pred.cpu().numpy())
print("Saved style_depth.npy")

# 5. Save preview (optional)
depth_colored = colorize(pred).convert("RGB")
depth_colored.save("style_depth_preview.png")
print("Saved style_depth_preview.png")
