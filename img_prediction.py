import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained CNN model
model = load_model('./AP_model/model_cnn.h5')

def preprocess_image(image_path):
    # Read and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
    img = cv2.resize(img, (100, 100))  # Resize to match input shape of the model
    img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add batch dimension
    img = np.expand_dims(img, axis=0)  # Add channel dimension for grayscale
    return img

def predict_age(image_path):
    # Preprocess the image
    img = preprocess_image(image_path)
    
    # Predict age using the model
    predicted_age = model.predict(img)[0][0]
    return predicted_age

# Test the prediction function with an image
image_path = './test_images/04.jpg'
predicted_age = predict_age(image_path)
print(f'Predicted age: {predicted_age}')
