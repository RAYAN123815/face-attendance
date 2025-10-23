# face_attendance_streamlit.py
# ----------------------------------------------------------
# Real-Time Face Recognition Attendance System (Streamlit, DeepFace)
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
st.markdown("This app detects faces in real time and marks attendance automatically.")

# Directories and files
KNOWN_FACES_DIR = "known_faces"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time", "Status"]).to_csv(ATTENDANCE_FILE, index=False)

# -------------------- LOAD KNOWN FACES --------------------
known_face_paths = [os.path.join(KNOWN_FACES_DIR, f) for f in os.listdir(KNOWN_FACES_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))]
known_face_names = [os.path.splitext(os.path.basename(f))[0] for f in known_face_paths]

# -------------------- FUNCTIONS --------------------
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.loc[len(df)] = [name, time_now, "Present"]
    df.to_csv(ATTENDANCE_FILE, index=False)

def register_new_face(name):
    cap = cv2.VideoCapture(0)
    st.info("📸 Press 'Spacebar' to capture face, 'Esc' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Register Face - Press Spacebar", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(file_path, frame)
            st.success(f"✅ Face saved as {file_path}")
            break
    cap.release()
    cv2.destroyAllWindows()

# -------------------- SIDEBAR --------------------
st.sidebar.header("🧠 Controls")
mode = st.sidebar.selectbox("Select Mode", ["Recognize Faces", "Register New Face", "View Attendance"])
st.sidebar.markdown("---")

# -------------------- MODE 1: RECOGNITION --------------------
if mode == "Recognize Faces":
    st.subheader("🟢 Real-Time Recognition")
    run = st.checkbox("Start Camera")

    if run:
        camera = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])
        attendance_set = set()

        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("Camera not accessible.")
                break

            for known_path, name in zip(known_face_paths, known_face_names):
                try:
                    result = DeepFace.verify(frame, known_path, model_name="VGG-Face", enforce_detection=False)
                    if result["verified"]:
                        if name not in attendance_set:
                            mark_attendance(name)
                            attendance_set.add(name)
                        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        break
                except Exception as e:
                    pass

            FRAME_WINDOW.image(frame, channels="BGR")

        camera.release()

# -------------------- MODE 2: REGISTER NEW FACE --------------------
elif mode == "Register New Face":
    st.subheader("📷 Register a New Face")
    new_name = st.text_input("Enter Name:")
    if st.button("Capt
