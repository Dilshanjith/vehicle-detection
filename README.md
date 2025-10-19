🚗 Vehicle Detection using CNN (TensorFlow & Keras)

This project uses a Convolutional Neural Network (CNN) to classify images as vehicle or non-vehicle.
It is built using TensorFlow and Keras, with configurable settings in config.

📂 Project Structure

vehicle-detection/
│
├── config.py             # Configuration file for dataset and model settings
├── train_model.py        # Script to train the CNN model
├── predict.py            # Script to predict a single image
├── data/                 # Dataset folder (contains 'vehicles' and 'non-vehicles')
│   ├── vehicles/
│   └── non-vehicles/
├── vehicle_cnn_model.h5  # Saved model after training
├── class_names.pkl       # Saved class label mapping (auto-generated after training)
└── README.md             # Project documentation


🧩 Dataset

data/
│
├── vehicles/
│   ├── car1.jpg
│   ├── truck2.png
│   └── ...
│
└── non-vehicles/
    ├── tree1.jpg
    ├── road2.jpg
    └── ...

⚙️ Configuration (config.py)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = 'vehicle_cnn_model.h5'
CLASS_NAMES = ['vehicle', 'non-vehicle']
