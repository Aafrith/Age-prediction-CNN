import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('./AP_model/model_cnn.h5')

# Load Haar Cascade classifier for face detection
haar = cv2.CascadeClassifier('./AP_model/haarcascade_frontalface_default.xml')

def preprocess_image(image):
    # Resize to match input shape of the model and normalize
    img = cv2.resize(image, (100, 100))
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension for grayscale
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def detect_faces(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect faces in the image
    faces = haar.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return img, faces

def predict_age(image_path):
    # Detect faces
    img, faces = detect_faces(image_path)
    
    # If no faces are detected, return None
    if len(faces) == 0:
        print("No faces detected.")
        return None
    
    # Predict age for each detected face
    ages = []
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]  # Extract face
        preprocessed_face = preprocess_image(face)
        predicted_age = model.predict(preprocessed_face)[0][0]
        ages.append((predicted_age, (x, y, w, h)))
    
    return img, ages

def display_faces_with_ages(image_path):
    # Predict ages and get faces
    img, predicted_ages = predict_age(image_path)
    
    if predicted_ages is not None:
        # Convert grayscale image to color for display purposes
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Draw rectangles and put predicted ages
        for idx, (age, (x, y, w, h)) in enumerate(predicted_ages):
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_color, f'Age: {age:.1f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the image with bounding boxes and predicted ages
        cv2.imshow('Faces with Predicted Ages', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No faces detected to display.")

# Test the display function with an image
image_path = './test_images/11.jpeg'
display_faces_with_ages(image_path)
