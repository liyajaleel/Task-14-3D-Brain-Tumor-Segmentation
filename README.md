# Task 14 – 3D Brain Tumor Segmentation using BraTS

## Objective
To build an end-to-end deep learning pipeline to segment brain tumor sub-regions
(edema, enhancing tumor, necrosis) from multi-modal 3D MRI scans.

## Dataset
BraTS 2020 / 2021 (Brain Tumor Segmentation Challenge)

Modalities:
- T1
- T1ce
- T2
- FLAIR

Note:
BraTS dataset requires registration and large downloads. The pipeline is implemented
to be fully compatible with BraTS MRI data. Due to the 1-day task timeline, sample
data is used to demonstrate the workflow.


## Phases Implemented
1. Data loading & visualization
2. 3D U-Net model definition
3. Training with Dice loss
4. Inference & visualization

## Visualization
A Jupyter Notebook (`visualization.ipynb`) is included to demonstrate
multi-modal MRI visualization, channel stacking, and normalization as
required in Phase 1 of the task.

## Model Architecture
A 3D U-Net model has been implemented using the MONAI framework as required in Phase 2
of the task. The model accepts 4-channel multi-modal MRI input (T1, T1ce, T2, FLAIR)
and outputs 3 segmentation classes corresponding to brain tumor sub-regions.





