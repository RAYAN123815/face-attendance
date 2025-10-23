import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Face Attendance", layout="wide")
st.title("📸 Face Recognition Attendance (DeepFace)")

KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [name, now]
    df.to_csv(ATTENDANCE_FILE, index=False)

def load_known_faces():
    paths = [os.path.join(KNOWN_FACES_DIR, f) for f in os.listdir(KNOWN_FACES_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    return paths, names

st.sidebar.header("🧠 Menu")
mode = st.sidebar.radio("Choose mode:", ["Register", "Recognize", "View Attendance"])

if mode == "Register":
    st.subheader("🧍 Register a New Face")
    name = st.text_input("Enter Name:")
    file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
    if st.button("Register"):
        if not name or not file:
            st.warning("Please provide both name and image.")
        else:
            img_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            with open(img_path, "wb") as f:
                f.write(file.read())
            st.success(f"✅ {name} registered successfully!")

elif mode == "Recognize":
    st.subheader("🔍 Recognize and Mark Attendance")
    file = st.file_uploader("Upload image for recognition", type=["jpg", "jpeg", "png"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
        st.image(img, channels="BGR", caption="Uploaded Image")

        known_paths, known_names = load_known_faces()
        match_found = None

        for path, name in zip(known_paths, known_names):
            try:
                result = DeepFace.verify(img, path, model_name="VGG-Face", enforce_detection=False)
                if result["verified"]:
                    match_found = name
                    mark_attendance(name)
                    break
            except Exception:
                continue

        if match_found:
            st.success(f"✅ Recognized: {match_found}")
        else:
            st.error("❌ No match found")

elif mode == "View Attendance":
    st.subheader("📋 Attendance Records")
    df = pd.read_csv(ATTENDANCE_FILE)
    st.dataframe(df[::-1], use_container_width=True)
