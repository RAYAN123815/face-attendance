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

    uploaded_file = st.file_uploader("📸 Upload a photo for recognition", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, channels="BGR", caption="Uploaded Image")

    matched_name = None
    for known_path, name in zip(known_face_paths, known_face_names):
        result = DeepFace.verify(image, known_path, model_name="VGG-Face", enforce_detection=False)
        if result["verified"]:
            matched_name = name
            mark_attendance(name)
            break

    if matched_name:
        st.success(f"✅ Recognized: {matched_name}")
    else:
        st.error("❌ No match found.")

    # Optional upload fallback for Streamlit Cloud
    uploaded_file = st.file_uploader("Or upload an image for recognition", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        detected = False
        for known_path, name in zip(known_face_paths, known_face_names):
            try:
                result = DeepFace.verify(frame, known_path, model_name="VGG-Face", enforce_detection=False)
                if result["verified"]:
                    mark_attendance(name)
                    st.success(f"✅ Recognized: {name}")
                    detected = True
                    break
            except Exception:
                pass
        if not detected:
            st.warning("No known face detected in uploaded image.")


# -------------------- MODE 2: REGISTER NEW FACE --------------------
elif mode == "Register New Face":
    st.subheader("📷 Register a New Face")
    new_name = st.text_input("Enter Name:")
    if st.button("Capture Face"):
        if new_name.strip() == "":
            st.warning("Please enter a valid name first.")
        else:
            register_new_face(new_name)
            st.success(f"Face for {new_name} registered successfully. Restart app to refresh database.")

# -------------------- MODE 3: ATTENDANCE TABLE --------------------
elif mode == "View Attendance":
    st.subheader("📄 Attendance Records")
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        st.dataframe(df[::-1], use_container_width=True)
    else:
        st.info("No attendance records found yet.")
