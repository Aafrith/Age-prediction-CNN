import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('./AP_model/model_cnn.h5')

def preprocess_frame(frame):
    # Preprocess the frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    frame_resized = cv2.resize(frame_gray, (100, 100))  # Resize to match input shape of the model
    frame_normalized = frame_resized.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    frame_expanded = np.expand_dims(frame_normalized, axis=-1)  # Add batch dimension
    frame_expanded = np.expand_dims(frame_expanded, axis=0)  # Add channel dimension for grayscale
    return frame_expanded

def predict_age_from_frame(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)
    
    # Predict age using the model
    predicted_age = model.predict(preprocessed_frame)[0][0]
    return predicted_age

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    
    # Predict age from the frame
    predicted_age = predict_age_from_frame(frame)
    
    # Display the predicted age on the frame
    cv2.putText(frame, f'Predicted Age: {predicted_age:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Live Streaming', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
