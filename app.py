import streamlit as st
import face_recognition
import numpy as np
import pandas as pd
import cv2
from datetime import datetime
import os

st.set_page_config(page_title="Face Attendance System", layout="wide")
st.title("📸 Face Attendance System")

# Folders
os.makedirs("registered_faces", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

ATTENDANCE_FILE = "attendance.csv"

# Load existing encodings
known_encodings = []
known_names = []

def load_registered_faces():
    known_encodings.clear()
    known_names.clear()
    for name in os.listdir("registered_faces"):
        path = os.path.join("registered_faces", name)
        if os.path.isfile(path):
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(name)[0])

load_registered_faces()

# Initialize CSV
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# --- Register new face ---
st.header("🧍 Register New Face")
with st.form("register_form"):
    name = st.text_input("Enter Name")
    img_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
    submit_reg = st.form_submit_button("Register")

    if submit_reg and name and img_file:
        img_path = os.path.join("registered_faces", f"{name}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_file.read())
        st.success(f"{name} registered successfully ✅")
        load_registered_faces()

# --- Recognize faces ---
st.header("🔍 Recognize Faces for Attendance")
recog_file = st.file_uploader("Upload Image for Attendance", type=["jpg", "jpeg", "png"])

if recog_file:
    image_path = os.path.join("uploads", recog_file.name)
    with open(image_path, "wb") as f:
        f.write(recog_file.read())

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    attendance = pd.read_csv(ATTENDANCE_FILE)
    found_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        found_names.append(name)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
        cv2.putText(image_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if name != "Unknown":
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name not in attendance["Name"].values:
                attendance.loc[len(attendance)] = [name, now]

    attendance.to_csv(ATTENDANCE_FILE, index=False)
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)
    st.write("✅ Attendance Recorded for:", ", ".join(found_names))
    st.dataframe(attendance)
