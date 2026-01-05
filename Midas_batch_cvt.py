import cv2
import torch
import os
import numpy as np
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

# === Configuration ===
input_folder = "assets"  # Folder containing input images
output_folder = "assets/depth"  # Folder to save processed depth images
os.makedirs(output_folder, exist_ok=True)  # Create output folder if not exists

# === Load MiDaS Model ===
# Choose from:
model_type = "DPT_Large" 
#model_type = "DPT_Hybrid"
#model_type = "MiDaS_small"

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# === Load MiDaS Transforms ===
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# === Process Images in Batch ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"depth_{filename}")

        # Load and preprocess the image
        img = cv2.imread(input_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        input_batch = transform(img_rgb).to(device)  # Apply MiDaS preprocessing

        # Predict depth
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],  # Resize to match original image size
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize depth map for visualization
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Save depth image
        cv2.imwrite(output_path, depth_map)
        print(f"Saved depth image: {output_path}")

print("Batch processing complete!")
