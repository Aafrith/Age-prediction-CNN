import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained CNN model
model = load_model('./AP_model/classification_model_cnn.h5')

# Load Haar Cascade classifier for face detection
haar = cv2.CascadeClassifier('./AP_model/haarcascade_frontalface_default.xml')

# Load the label encoder
with open('./AP_data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def preprocess_face(face):
    # Resize to match input shape of the model and normalize
    face_resized = cv2.resize(face, (100, 100))
    face_normalized = face_resized.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    face_expanded = np.expand_dims(face_normalized, axis=-1)  # Add channel dimension for grayscale
    face_expanded = np.expand_dims(face_expanded, axis=0)  # Add batch dimension
    return face_expanded

def predict_age_from_face(face):
    # Preprocess the face
    preprocessed_face = preprocess_face(face)
    
    # Predict age using the model
    prediction = model.predict(preprocessed_face)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_age_range = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_age_range

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = haar.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face = gray_frame[y:y+h, x:x+w]
        
        # Predict age from the face
        predicted_age_range = predict_age_from_face(face)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display the predicted age on the frame
        cv2.putText(frame, f'Age: {predicted_age_range}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow('Live Streaming', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
