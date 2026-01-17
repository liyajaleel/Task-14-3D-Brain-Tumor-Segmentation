"""
model.py

This file defines the 3D U-Net model architecture for
brain tumor segmentation as specified in Task 14.

The model is implemented using the MONAI framework.
"""

from monai.networks.nets import UNet


def get_model():
    """
    Creates and returns a 3D U-Net model.

    Input:
        - 4 channels (T1, T1ce, T2, FLAIR)

    Output:
        - 3 segmentation classes
          (edema, enhancing tumor, necrosis)
    """

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
    )

    return model
