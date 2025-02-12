import cv2
from keras.models import load_model
import numpy as np
import pygame  # For playing music
import os
import time  # For delay implementation

# Force Pygame to use a compatible audio driver
os.environ['SDL_AUDIODRIVER'] = 'directsound'

# Load the trained model
model = load_model('emotion_model.h5')

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Map emotions to specific music file paths
music_files = {
    'angry': r'C:\Users\Aditya\Music\music_from_envy\Angry_Song.mp3',
    'disgust': r'C:\Users\Aditya\Music\music_from_envy\Disgust_Song.mp3',
    'fear': r'C:\Users\Aditya\Music\music_from_envy\Fear_Song.mp3',
    'happy': r'C:\Users\Aditya\Music\music_from_envy\Pharrell Williams - Happy.mp3',
    'sad': r'C:\Users\Aditya\Music\music_from_envy\Sad_Song.mp3',
    'surprise': r'C:\Users\Aditya\Music\music_from_envy\Surprise_Song.mp3',
    'neutral': r'C:\Users\Aditya\Music\music_from_envy\Neutral_Song.mp3'
}

# Initialize Pygame for playing music
pygame.mixer.init()

# Start webcam
cap = cv2.VideoCapture(0)

def preprocess_frame(frame):
    """Preprocess the face region to match the model's input format."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face = cv2.resize(gray_frame, (48, 48))  # Resize to 48x48
    face = np.reshape(face, (1, 48, 48, 1)) / 255.0  # Normalize pixel values
    return face

emotion_detected = False  # Flag to stop the application after detecting an emotion
last_detection_time = time.time()  # Track the time of the last detection attempt

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Check if 7 seconds have passed since the last detection
    current_time = time.time()
    if current_time - last_detection_time >= 7:  # Minimum 7-second delay
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Preprocess face region for prediction
            face = preprocess_frame(frame[y:y+h, x:x+w])
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]  # Get predicted emotion

            # Display the emotion on the video feed
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Play music and stop after detecting the emotion
            if not emotion_detected:
                print(f"Detected Emotion: {emotion}")
                if emotion in music_files:
                    try:
                        pygame.mixer.music.load(music_files[emotion])
                        pygame.mixer.music.play()
                        print(f"Playing: {music_files[emotion]}")
                    except pygame.error as e:
                        print(f"Error playing music: {e}")
                else:
                    print(f"No music file found for emotion: {emotion}")
                emotion_detected = True
                break

        # Stop the webcam feed after detecting the emotion
        if emotion_detected:
            print("Stopping webcam feed...")
            break

    # Show the video feed
    cv2.imshow('Emotion Recognition', frame)

    # Add controls for pausing, resuming, and stopping the music
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit application
        break
    elif key == ord('p'):  # Pause music
        pygame.mixer.music.pause()
        print("Music paused")
    elif key == ord('r'):  # Resume music
        pygame.mixer.music.unpause()
        print("Music resumed")

# Release resources and stop the webcam
cap.release()
cv2.destroyAllWindows()

# Wait for the song to finish playing or quit immediately
if emotion_detected:
    while pygame.mixer.music.get_busy():
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit application and stop music
            pygame.mixer.music.stop()
            print("Music stopped")
            break
pygame.mixer.quit()
