import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH

# --- Load dataset with training/validation split ---
train_dataset = image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = image_dataset_from_directory(
    DATA_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# ✅ Save class names
class_names = train_dataset.class_names
print("Detected classes:", class_names)

# --- Normalize images ---
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))

# --- Build CNN model ---
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# --- Compile model ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train model ---
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# --- Save model ---
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved at {MODEL_SAVE_PATH}")
