import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from deepface import DeepFace

# -------------------- SETUP --------------------
st.set_page_config(page_title="🎥 Face Verification", layout="wide")
st.title("🎓 Face Verification System (Webcam + DeepFace)")
st.markdown("Verify faces via webcam or upload. Works locally and in the cloud!")

# Directories
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# -------------------- FUNCTIONS --------------------
def save_uploaded_face(uploaded_file, name):
    path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    cv2.imwrite(path, image)
    return path

def load_known_faces():
    paths = [os.path.join(KNOWN_FACES_DIR, f) for f in os.listdir(KNOWN_FACES_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    names = [os.path.splitext(os.path.basename(f))[0] for f in paths]
    return paths, names

def verify_face(face_img, known_face_paths, known_face_names):
    for path, name in zip(known_face_paths, known_face_names):
        try:
            result = DeepFace.verify(face_img, path, model_name="VGG-Face", enforce_detection=False)
            if result["verified"]:
                return name
        except Exception:
            continue
    return None

# -------------------- SIDEBAR --------------------
st.sidebar.header("🧠 Controls")
mode = st.sidebar.radio("Select Mode", ["Webcam Verification", "Upload Verification", "Register Face"])

known_face_paths, known_face_names = load_known_faces()

# -------------------- REGISTER FACE --------------------
if mode == "Register Face":
    st.subheader("📷 Register a New Face")
    name = st.text_input("Enter Name:")
    uploaded = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])

    if st.button("Register"):
        if not name.strip():
            st.warning("Please enter a valid name.")
        elif uploaded is None:
            st.warning("Please upload a photo.")
        else:
            save_uploaded_face(uploaded, name)
            st.success(f"✅ Face for '{name}' registered successfully!")

# -------------------- UPLOAD VERIFICATION --------------------
elif mode == "Upload Verification":
    st.subheader("📁 Upload a Photo to Verify")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
        st.image(img, channels="BGR", caption="Uploaded Image")

        match = verify_face(img, known_face_paths, known_face_names)
        if match:
            st.success(f"✅ Match found: {match}")
        else:
            st.error("❌ No match found.")

# -------------------- WEBCAM VERIFICATION --------------------
elif mode == "Webcam Verification":
    st.subheader("🎥 Webcam Face Verification")

    # Try to open camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("⚠️ Webcam not accessible. Try running locally with: `streamlit run app.py`")
    else:
        FRAME_WINDOW = st.image([])
        run = st.checkbox("Start Webcam")

        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("Camera frame not available.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            # Try verifying every frame (you can make this button-triggered too)
            match = verify_face(frame_rgb, known_face_paths, known_face_names)
            if match:
                st.success(f"✅ Match found: {match}")
                break

        camera.release()
        st.info("Webcam stopped.")
