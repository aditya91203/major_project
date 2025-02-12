import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the path to the dataset
train_path = r"C:\Users\Aditya\Downloads\train"
test_path = r"C:\Users\Aditya\Downloads\test"

# Map emotions to numeric labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
emotion_mapping = {emotion.lower(): idx for idx, emotion in enumerate(emotion_labels)}

# Function to load images and labels
def load_data(data_path):
    images, labels = [], []
    for emotion_folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, emotion_folder)
        if os.path.isdir(folder_path):
            label = emotion_mapping[emotion_folder.lower()]  # Convert folder name to lowercase
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (48, 48))  # Resize to 48x48
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Load training and testing data
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

# Normalize image data
X_train = X_train.reshape(-1, 48, 48, 1) / 255.0
X_test = X_test.reshape(-1, 48, 48, 1) / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion categories
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_data=(X_val, y_val))

# Save the trained model
model.save('emotion_model.h5')
print("Model saved as emotion_model.h5")

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
