import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
from PIL import Image

# Load the trained CNN model
model = load_model('./AP_model/classification_model_cnn.h5')

# Load Haar Cascade classifier for face detection
haar = cv2.CascadeClassifier('./AP_model/haarcascade_frontalface_default.xml')

# Load the label encoder
with open('./AP_data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

def preprocess_face(face):
    face_resized = cv2.resize(face, (100, 100))
    face_normalized = face_resized.astype('float32') / 255.0
    face_expanded = np.expand_dims(face_normalized, axis=-1)
    face_expanded = np.expand_dims(face_expanded, axis=0)
    return face_expanded

def predict_age_from_face(face):
    preprocessed_face = preprocess_face(face)
    prediction = model.predict(preprocessed_face)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_age_range = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_age_range

def browse_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = np.array(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        st.write("No faces detected.")
        return
    
    x, y, w, h = faces[0]
    face = gray_img[y:y+h, x:x+w]
    age_range = predict_age_from_face(face)
    
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img, f'Age: {age_range}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    st.image(img, caption=f"Predicted Age: {age_range}", use_column_width=True)

def live_prediction():
    cap = cv2.VideoCapture(0)
    
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = gray_frame[y:y+h, x:x+w]
            age_range = predict_age_from_face(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Age: {age_range}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

# Custom CSS for styling
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
        padding: 2rem;
    }
    .sidebar .sidebar-content {
        background: #404040;
        color: white;
    }
    .css-1aumxhk {
        color: #404040;
        font-size: 1.5rem;
    }
    .css-1vbd788 {
        font-size: 1rem;
        color: #404040;
    }
    </style>
    """, unsafe_allow_html=True)

# Create the Streamlit UI
st.title("Age Prediction")
st.write("---")

st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Image Prediction", "Live Prediction"])

if app_mode == "Image Prediction":
    st.header("Browse Image")
    st.write("Upload an image to predict the age range of the detected face.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        browse_image(uploaded_file)

elif app_mode == "Live Prediction":
    st.header("Live Prediction")
    st.write("Start your webcam to predict the age range of the detected faces in real-time.")
    live_prediction()
