import streamlit as st
import cv2
import numpy as np
import os
from deepface import DeepFace

# -------------------- FACE DETECTION --------------------
def is_face_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def verify_face_image(img_path):
    # Step 1: Check if a face exists
    if not is_face_image(img_path):
        return False, "❌ No valid human face detected."

    # Step 2: Try to analyze with DeepFace
    try:
        DeepFace.represent(img_path, model_name="Facenet")
        return True, "✅ Human face detected and verified."
    except Exception:
        return False, "❌ Invalid or non-human face detected."

# -------------------- APP SETUP --------------------
st.set_page_config(page_title="🎥 Smart Face Verification", layout="centered")
st.title("🎓 Smart Face Verification System")
st.markdown("This app verifies **real human faces only** using AI. Upload or capture your image below.")

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# -------------------- UTILITIES --------------------
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

def verify_face(img_path, known_face_paths, known_face_names):
    for path, name in zip(known_face_paths, known_face_names):
        try:
            result = DeepFace.verify(img_path, path, model_name="VGG-Face", enforce_detection=False)
            if result["verified"]:
                return name
        except Exception:
            continue
    return None

# -------------------- SIDEBAR --------------------
st.sidebar.header("🧠 Controls")
mode = st.sidebar.radio("Select Mode", ["Register Face", "Upload Verification", "Webcam Verification"])

known_face_paths, known_face_names = load_known_faces()

# -------------------- REGISTER FACE --------------------
if mode == "Register Face":
    st.subheader("📷 Register a New Face")
    name = st.text_input("Enter Name:")
    uploaded = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])

    if st.button("Register Face"):
        if not name.strip():
            st.warning("⚠️ Please enter a valid name.")
        elif uploaded is None:
            st.warning("⚠️ Please upload a photo.")
        else:
            with open("temp.jpg", "wb") as f:
                f.write(uploaded.getbuffer())
            valid, msg = verify_face_image("temp.jpg")
            if not valid:
                st.error(msg)
            else:
                save_uploaded_face(uploaded, name)
                st.success(f"✅ Face for '{name}' registered successfully!")

# -------------------- UPLOAD VERIFICATION --------------------
elif mode == "Upload Verification":
    st.subheader("📁 Upload a Photo to Verify")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded:
        with open("temp.jpg", "wb") as f:
            f.write(uploaded.getbuffer())

        st.image("temp.jpg", caption="Uploaded Image", use_container_width=True)

        valid, msg = verify_face_image("temp.jpg")
        if not valid:
            st.error(msg)
        else:
            match = verify_face("temp.jpg", known_face_paths, known_face_names)
            if match:
                st.success(f"✅ Match found: {match}")
            else:
                st.warning("❌ No match found in the database.")

# -------------------- WEBCAM VERIFICATION --------------------
elif mode == "Webcam Verification":
    st.subheader("🎥 Webcam Face Verification")

    img_file_buffer = st.camera_input("Take a photo")

    if img_file_buffer is not None:
        with open("temp.jpg", "wb") as f:
            f.write(img_file_buffer.getbuffer())

        st.image("temp.jpg", caption="Captured Image", use_container_width=True)

        valid, msg = verify_face_image("temp.jpg")
        if not valid:
            st.error(msg)
        else:
            match = verify_face("temp.jpg", known_face_paths, known_face_names)
            if match:
                st.success(f"✅ Match found: {match}")
            else:
                st.warning("❌ No match found in the database.")
