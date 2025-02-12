#2
#preprocessing
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Path to dataset
dataset_path = r"C:\Users\Aditya\Downloads\train"

# Emotion labels based on folder names
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_mapping = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

# Function to load images and labels
def load_data(data_path):
    images, labels = [], []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):  # Ensure it's a directory
            label = emotion_mapping[folder]  # Map folder name to label
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image
                if img is not None:
                    img = cv2.resize(img, (48, 48))  # Resize to 48x48
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess the dataset
X, y = load_data(dataset_path)

# Normalize image data and reshape
X = X.reshape(-1, 48, 48, 1) / 255.0

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=len(emotion_labels))

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
