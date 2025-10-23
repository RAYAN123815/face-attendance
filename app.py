import streamlit as st
import numpy as np
import os

# Try importing cv2 safely
try:
    import cv2
except ImportError:
    st.error("⚠️ OpenCV failed to load. Please ensure opencv-python-headless is installed.")
    st.stop()

from deepface import DeepFace

st.set_page_config(page_title="Face Verification", layout="centered")
st.title("🎯 Simple Face Verification")
st.markdown("Upload an image — this app checks if it’s a **real human face**.")

def is_face_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def verify_face_image(img_array):
    if not is_face_image(img_array):
        return False, "❌ No human face detected."

    try:
        DeepFace.represent(img_array, model_name="Facenet", enforce_detection=False)
        return True, "✅ Human face verified successfully!"
    except Exception:
        return False, "⚠️ Could not verify face. Try a clearer photo."

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded Image")

    valid, msg = verify_face_image(img)
    if valid:
        st.success(msg)
    else:
        st.error(msg)
