# Image Mosaic Generator - Final Project for Stats507

## Overview

This project implements an image mosaic generator that reconstructs a main image using small tile images from a gallery dataset.  
It supports multiple feature extraction methods (color histograms, shallow CNN features, HOG descriptors, and edge density), and provides a modular pipeline for tiling, feature matching, image reconstruction, and evaluation.  
A weight tuning system is also implemented to improve reconstruction quality using both visual and quantitative metrics.

## Directory Structure and Key Files

- `mosaic_pipeline.py`  
  Main pipeline for image tiling and mosaic reconstruction. Includes modular extractors and matchers.

- `evaluator.py`  
  Evaluation metrics and weight tuning tools. Includes SSIM, PSNR, LPIPS, and grid search functions.

- `main_image_tile_features.pkl`  
  Cached features of the main image’s divided tiles.

- `features_combined.pkl`, `matched_tile_to_gallery_faiss.pkl`  
  Intermediate results for FAISS-based tile-to-gallery image matching.

- `*.jpg`, `*.JPEG`  
  Images used for either the target image or gallery tiles.

- `tiny-imagenet-200/` (sample only)  
  A **small subset** of the Tiny ImageNet dataset is included for demonstration purposes.  
  The **full dataset** is too large to upload to GitHub and must be downloaded manually.


- `*.ipynb`  
  Jupyter notebooks for experiments and demonstration:
  - `Experiment.ipynb`: Visual and quantitative model comparisons.
  - `image prepare.ipynb`: Preprocessing the gallery dataset.
  - `Model tunning.ipynb`: Hyperparameter search and SSIM-based evaluation.
  - `Full pipeline example.ipynb`: End-to-end pipeline demo on sample image.
 
## Dataset

Due to GitHub file size limitations, only a **trimmed-down (partial)** version of the Tiny ImageNet dataset is included in this repository.  
To achieve full performance and reproduce all experiments, please download the **complete dataset** from Kaggle:

🔗 [https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200](https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200)

After downloading, extract the dataset into the project root so that the directory `tiny-imagenet-200/` is available.

---


## How to Run

This project is designed to be executed step-by-step using the provided Jupyter notebooks.  
Start from `image prepare.ipynb`, then proceed to feature extraction, matching, and finally reconstruction and evaluation.
