# ============================================================
# config.py — Configuration file for Vehicle Detection Project
# ============================================================

import os
import kagglehub

# ------------------------------------------------------------
# 🧩 Dataset Configuration
# ------------------------------------------------------------

# Automatically download the latest dataset from KaggleHub
DATASET_PATH = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")

# You can also specify a custom dataset folder (optional)
DATA_DIR = DATASET_PATH   # Use the path KaggleHub returns

# If you’ve extracted or organized folders manually:
DATA_DIR = './data'

TRAINING_DIR = os.path.join(DATA_DIR, 'non-vehicles')   # Change based on your dataset structure
TEST_DIR = os.path.join(DATA_DIR, 'vehicles')           # Change based on your dataset structure

# ------------------------------------------------------------
# ⚙️ Model Configuration
# ------------------------------------------------------------
MODEL_SAVE_PATH = 'vehicle_cnn_model.h5'   # File name for saving the trained model
MODEL_PATH = 'vehicle_cnn_model.h5'


IMAGE_SIZE = (128, 128)     # Resize all images to this size
BATCH_SIZE = 32             # Number of samples per batch
EPOCHS = 15                 # Number of epochs for training



CLASS_NAMES = ['non-vehical','vehical']



SEED = 123
