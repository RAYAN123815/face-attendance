# face_attendance_streamlit.py
# ----------------------------------------------------------
# Real-Time Face Recognition Attendance System (Streamlit)
# ----------------------------------------------------------

import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import dlib

print("dlib version:", dlib.__version__)
print("Face recognition imported successfully ✅")


# -------------------- SETUP --------------------
st.set_page_config(page_title="🎥 Face Attendance System", layout="wide")
st.title("🎓 Face Recognition Attendance System")
st.markdown("This app detects faces in real time and marks attendance automatically.")

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Attendance CSV
ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Time", "Status"]).to_csv(ATTENDANCE_FILE, index=False)

# -------------------- LOAD KNOWN FACES --------------------
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, filename))
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

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
        if key == 27:  # ESC
            break
        elif key == 32:  # Spacebar
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
        attendance_set = set()

        FRAME_WINDOW = st.image([])

        while run:
            ret, frame = camera.read()
            if not ret:
                st.warning("Camera not accessible.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

                if best_match_index is not None and matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name not in attendance_set:
                        mark_attendance(name)
                        attendance_set.add(name)

                # Draw rectangle and label
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            FRAME_WINDOW.image(frame, channels="BGR")

        camera.release()

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