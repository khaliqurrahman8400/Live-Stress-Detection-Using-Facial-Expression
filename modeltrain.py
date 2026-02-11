import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directories
train_dir = r"C:\Users\user\Documents\Stress Detection Using Facial Expression\train"
test_dir = r"C:\Users\user\Documents\Stress Detection Using Facial Expression\test"

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load dataset
train_data = train_datagen.flow_from_directory(train_dir, target_size=(48, 48), batch_size=32, class_mode='categorical', color_mode="grayscale")
test_data = test_datagen.flow_from_directory(test_dir, target_size=(48, 48), batch_size=32, class_mode='categorical', color_mode="grayscale")

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')  # Output layer
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_data, epochs=20, validation_data=test_data)

# Save the trained model
model.save("stress_detection_model.h5")
print("Model saved successfully!")