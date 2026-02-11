import cv2
import os
import numpy as np

# Function to preprocess an image
def preprocess_image(image_path, target_size=(48, 48)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"Warning: Unable to read {image_path}")
        return None
    
    img = cv2.resize(img, target_size)  # Resize
    img = img.astype(np.float32) / 255.0  # Normalize pixel values (0-1)
    return img

# Function to preprocess all images in a dataset folder
def preprocess_dataset(dataset_path):
    images = []
    labels = []
    class_names = os.listdir(dataset_path)  # Get subdirectories (class names)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        
        if not os.path.isdir(class_path):  # Skip if not a directory
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            processed_img = preprocess_image(img_path)
            if processed_img is not None:
                images.append(processed_img)
                labels.append(label)  # Assign numerical label
    
    return np.array(images), np.array(labels)

# ✅ Define dataset paths (Use raw string format r"" to avoid path issues)
train_path =  "C:\\Users\\user\\Documents\\Stress Detection Using Facial Expression\\train"
test_path = "C:\\Users\\user\\Documents\\Stress Detection Using Facial Expression\\test"

# ✅ Preprocess entire dataset
train_images, train_labels = preprocess_dataset(train_path)
test_images, test_labels = preprocess_dataset(test_path)

# ✅ Print dataset sizes
print("Train dataset size:", train_images.shape, train_labels.shape)
print("Test dataset size:", test_images.shape, test_labels.shape)