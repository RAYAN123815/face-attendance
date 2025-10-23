# face_attendance_streamlit.py
# ----------------------------------------------------------
# Face Recognition Attendance System (Streamlit + DeepFace)
# Works on Streamlit Cloud using image uploads
# ----------------------------------------------------------

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from deepface import DeepFace

# -------------------- SETUP --------------------
st.set_page_config(page_title="🎥 Face Attendance System", layout="wide")
st.title("🎓 Face Recognition Attendance System (DeepFace)")
st.markdown("Upload photos to register or recognize faces. Works fully in Streamlit Cloud!")

# Directories and files
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time", "Status"]).to_csv(ATTENDANCE_FILE, index=False)

# -------------------- LOAD KNOWN FACES --------------------
def load_known_faces():
    known_paths = [os.path.join(KNOWN_FACES_DIR, f) for f in os.listdir(KNOWN_FACES_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    known_names = [os.path.splitext(os.path.basename(f))[0] for f in known_paths]
    return known_paths, known_names

known_face_paths, known_face_names = load_known_faces()

# -------------------- FUNCTIONS --------------------
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [name, time_now, "Present"]
    df.to_csv(ATTENDANCE_FILE, index=False)

def save_uploaded_face(uploaded_file, name):
    """Save uploaded face image under known_faces"""
    file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    cv2.imwrite(file_path, image)
    return file_path

# -------------------- SIDEBAR --------------------
st.sidebar.header("🧠 Controls")
mode = st.sidebar.selectbox("Select Mode", ["Recognize Faces", "Register New Face", "View Attendance"])
st.sidebar.markdown("---")

# -------------------- MODE 1: RECOGNIZE --------------------
if mode == "Recognize Faces":
    st.subheader("🟢 Face Recognition via Uploaded Image")

    uploaded_file = st.file_uploader("📸 Upload an image to recognize", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        st.image(image, channels="BGR", caption="Uploaded Image")

        matched_name = None
        for known_path, name in zip(known_face_paths, known_face_names):
            try:
                result = DeepFace.verify(image, known_path, model_name="VGG-Face", enforce_detection=False)
                if result["verified"]:
                    matched_name = name
                    mark_attendance(name)
                    break
            except Exception:
                continue

        if matched_name:
            st.success(f"✅ Recognized: {matched_name}")
        else:
            st.error("❌ No match found in database.")

# -------------------- MODE 2: REGISTER --------------------
elif mode == "Register New Face":
    st.subheader("📷 Register a New Face")
    new_name = st.text_input("Enter Name:")
    uploaded_file = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])

    if st.button("Register Face"):
        if new_name.strip() == "":
            st.warning("Please enter a valid name first.")
        elif uploaded_file is None:
            st.warning("Please upload an image.")
        else:
            save_uploaded_face(uploaded_file, new_name)
            st.success(f"✅ Face for '{new_name}' registered successfully!")
            known_face_paths, known_face_names = load_known_faces()

# -------------------- MODE 3: ATTENDANCE --------------------
elif mode == "View Attendance":
    st.subheader("📄 Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df[::-1], use_container_width=True)
    else:
        st.info("No attendance records found yet.")
