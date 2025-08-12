import cv2
import mediapipe as mp
import numpy as np
import threading
import pygame
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings

warnings.filterwarnings("ignore")

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Ensure this file exists in your directory

# Flag to avoid overlapping alerts
alert_playing = False

# Flag for email sent
email_sent = False

# Email setup
def send_email():
    global email_sent
    if not email_sent:
        email_sent = True
        
        # Set up the server and sender's email
        sender_email = "brahmimaddineedi123@gmail.com"  # Replace with your email
        receiver_email = "kotanarasimhareddy9@gmail.com"  # Replace with recipient email
        password = "vpqt ffqi woru meju"  # Replace with your email password
        
        # Email content
        subject = "Drowsiness Alert!"
        body = "Drowsiness detected. Please take a break immediately."

        # Prepare the email
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            # Set up the SMTP server and send the email
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print("Email sent successfully!")
        except Exception as e:
            print(f"Error sending email: {e}")
            email_sent = False

# Play alert sound
def sound_alert():
    global alert_playing
    if not alert_playing:
        alert_playing = True
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
        alert_playing = False

# EAR calculation
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
    A = np.linalg.norm(points[1] - points[5])
    B = np.linalg.norm(points[2] - points[4])
    C = np.linalg.norm(points[0] - points[3])
    ear = (A + B) / (2.0 * C)
    return ear, points

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Eye landmark indices
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Thresholds
EAR_THRESHOLD = 0.20
CLOSED_FRAMES = 0

# Video stream
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
ALERT_FRAMES = int(fps * 3)  # 3 seconds of closed eyes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        left_ear, left_points = eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, w, h)
        right_ear, right_points = eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw eye points
        for pt in left_points + right_points:
            cv2.circle(frame, tuple(np.int32(pt)), 2, (0, 255, 0), -1)

        if avg_ear < EAR_THRESHOLD:
            CLOSED_FRAMES += 1
            if CLOSED_FRAMES == ALERT_FRAMES:
                threading.Thread(target=sound_alert).start()
                threading.Thread(target=send_email).start()  # Send email
        else:
            CLOSED_FRAMES = 0

        # Display EAR and warning if needed
        cv2.putText(frame, f'EAR: {avg_ear:.2f}', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        if CLOSED_FRAMES >= ALERT_FRAMES:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No face detected!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
