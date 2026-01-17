"""
train.py

Phase 3: Training & Validation (Task 14)

This script demonstrates a minimal training step for
3D Brain Tumor Segmentation using a 3D U-Net and Dice Loss.
"""

import torch
import numpy as np
from monai.losses import DiceLoss
from model import get_model

# Device (CPU is sufficient for demonstration)
device = torch.device("cpu")

# Create synthetic BraTS-style input data
# Shape: (batch, channels, height, width, depth)
image = np.random.rand(1, 4, 128, 128, 64)
label = np.random.randint(0, 3, size=(1, 128, 128, 64))

image = torch.tensor(image, dtype=torch.float32).to(device)
label = torch.tensor(label, dtype=torch.long).to(device)

# Load model
model = get_model().to(device)

# Define Dice Loss and optimizer
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Single training step
model.train()
optimizer.zero_grad()
outputs = model(image)
loss = loss_function(outputs, label)
loss.backward()
optimizer.step()

print("Training step completed successfully")
print("Dice Loss:", loss.item())
