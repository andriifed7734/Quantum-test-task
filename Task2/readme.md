# Task 2: Cross-Seasonal Sentinel-2 Image Matching

## Solution Overview
The objective of this task is to match cross-seasonal satellite imagery (Summer vs. Winter) from Sentinel-2. Since seasonal changes drastically alter the visual appearance of landscapes (e.g., snow cover, vegetation loss), traditional pixel-matching methods often fail. 

To solve this, the pipeline is divided into three stages:
1. **Data Preparation**: Raw Sentinel-2 L1C `.jp2` files are processed. A variance-based filtering system automatically discards homogeneous or "no-data" patches (like pure cloud cover), resulting in high-quality 512x512 training pairs.
2. **Model Training**: A deep learning approach using a **Siamese Network** with a ResNet18 backbone is trained to extract robust, season-invariant embeddings. 
3. **Inference & Matching**: The fine-tuned model compares patches to establish matches across drastic seasonal variations, outperforming standard baseline methods.

## Pre-trained Model & Raw Data
Due to size constraints, the raw Sentinel-2 satellite imagery and the pre-trained model weights are hosted externally on Google Drive with link below.
https://drive.google.com/drive/folders/1MF90YwOYPYiDkUtsVzj9sTdsSTjfNol7?usp=drive_link

Data Setup Instructions:
1. **Download the contents of the provided Google Drive link.**
2. **Place the pre-trained weights file (siamese_sentinel_weights.pth) directly into the root of this task folder (alongside the .ipynb files).**
3. **Create a data/ folder in this directory and place the downloaded Sentinel-2 folders (e.g., S2A_MSIL1C_20160618... and S2A_MSIL1C_20161205...) inside it. Ensure that the inner IMG_DATA folders containing the .jp2 files remain intact, as dataset_creation.ipynb will look for them.**

## Directory Structure
```text
task_2_/
│
├── dataset_creation.ipynb    # Notebook for extracting and filtering patches from raw Sentinel-2 data
├── train.py                  # Python script to train the Siamese Network
├── demo.ipynb                # Notebook for model inference and visual demonstration
├── requirements.txt          # Dependencies required for the task
└── README.md                 # This file
