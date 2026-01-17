"""
inference.py

Phase 4: Inference & Visualization (Task 14)

This script demonstrates inference using a trained 3D U-Net model
and visualizes the predicted segmentation mask on a sample MRI slice.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import get_model

# Load model
model = get_model()
model.eval()

# Create synthetic BraTS-style input
# Shape: (batch, channels, height, width, depth)
input_image = torch.rand(1, 4, 128, 128, 64)

# Run inference
with torch.no_grad():
    output = model(input_image)

# Convert prediction to segmentation mask
predicted_mask = torch.argmax(output, dim=1).squeeze().numpy()

# Visualize one slice
slice_index = 32

plt.figure(figsize=(5, 5))
plt.imshow(predicted_mask[:, :, slice_index], cmap="jet")
plt.title("Predicted Tumor Segmentation")
plt.axis("off")
plt.show()
