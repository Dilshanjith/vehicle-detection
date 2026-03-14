

import os
import kagglehub




DATASET_PATH = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")


DATA_DIR = DATASET_PATH    


DATA_DIR = './data'

TRAINING_DIR = os.path.join(DATA_DIR, 'non-vehicles')    
TEST_DIR = os.path.join(DATA_DIR, 'vehicles')           

MODEL_SAVE_PATH = 'vehicle_cnn_model.h5'    
MODEL_PATH = 'vehicle_cnn_model.h5'


IMAGE_SIZE = (128, 128)     
BATCH_SIZE = 32              
EPOCHS = 15                  


CLASS_NAMES = ['non-vehical','vehical']



SEED = 123
