# merging of face_api(student vector registeration) and pi_attendance(facial recognition)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from picamera2 import Picamera2
import cv2
import face_recognition
import chromadb
import time
import math
import mediapipe as mp
import atexit
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np

app = FastAPI()

# ✅ CORS Middleware (still needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use "*" temporarily
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Manual OPTIONS fallback
@app.options("/{full_path:path}")
async def preflight_handler(request: Request, full_path: str):
    return JSONResponse(
        status_code=204,
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )

# ────────────────────────────────
# ✅ ChromaDB
# ────────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="face_vector")

class RegisterFaceRequest(BaseModel):
    student_id: str
    name: str
    image_base64: str

@app.post("/register-face")
async def register_face(data: RegisterFaceRequest):
    try:
        image_data = base64.b64decode(data.image_base64.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        if len(face_encodings) == 1:
            vector = face_encodings[0].tolist()
            collection.add(
                embeddings=[vector],
                ids=[data.student_id],
                metadatas=[{"name": data.name}]
            )
            return JSONResponse(
                content={"status": "success", "message": f"Registered {data.name}"},
                headers={"Access-Control-Allow-Origin": "*"}
            )
        elif len(face_encodings) > 1:
            return JSONResponse(
                content={"status": "error", "message": "Multiple faces detected."},
                headers={"Access-Control-Allow-Origin": "*"}
            )
        else:
            return JSONResponse(
                content={"status": "error", "message": "No face detected."},
                headers={"Access-Control-Allow-Origin": "*"}
            )
    except Exception as e:
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    
# Camera Setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 1080)}))
picam2.start()

# Anti-Spoofing Setup
cv_scaler = 4
SPOOF_TIMEOUT = 10
EAR_THRESHOLD = 0.2
YAW_THRESHOLD = 10
PITCH_THRESHOLD = 10
last_blink_time = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks_2d, w, h):
    image_points = np.array([
        landmarks_2d[1],    # nose tip
        landmarks_2d[152],  # chin
        landmarks_2d[263],  # right eye corner
        landmarks_2d[33],   # left eye corner
        landmarks_2d[287],  # right mouth corner
        landmarks_2d[57],   # left mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (43.3, 32.7, -26.0),
        (-43.3, 32.7, -26.0),
        (28.9, -28.9, -24.1),
        (-28.9, -28.9, -24.1)
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def detect_spoofing(rgb_resized, blink_total):
    global last_blink_time
    results = face_mesh.process(rgb_resized)
    if not results.multi_face_landmarks:
        return blink_total, 0, 0

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = rgb_resized.shape
    landmarks_2d = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

    left_EAR = eye_aspect_ratio(landmarks_2d, LEFT_EYE_LANDMARKS)
    right_EAR = eye_aspect_ratio(landmarks_2d, RIGHT_EYE_LANDMARKS)
    avg_EAR = (left_EAR + right_EAR) / 2.0

    if avg_EAR < EAR_THRESHOLD:
        blink_total += 1
        last_blink_time = time.time()

    pitch, yaw, _ = get_head_pose(landmarks_2d, w, h)
    return blink_total, pitch, yaw

#connect this to teacherDashboard
@app.post("/start-attendance")
def start_attendance():
    recognized = []
    student_ids = set()
    blink_total = 0

    try:
        while True:
            frame = picam2.capture_array()
            resized = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
            rgb_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_resized)
            face_encodings = face_recognition.face_encodings(rgb_resized, face_locations, model='small')

            blink_total, pitch, yaw = detect_spoofing(rgb_resized, blink_total)

            if (time.time() - last_blink_time > SPOOF_TIMEOUT) or (abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD):
                continue  # spoof detected, skip frame

            for encoding in face_encodings:
                # find the top match
                matches = collection.query(query_embeddings=[encoding.tolist()], n_results=1)
                # see if the matches are close enough
                if matches["distances"] and matches["distances"][0][0] < 0.5:
                    student_id = matches["ids"][0][0]
                    if student_id not in student_ids: # check if the student is already in the attending list
                        student_ids.add(student_id)
                        recognized.append({
                            "student_id": student_id,
                            "timestamp": datetime.now().strftime("%H:%M:%S") #timestamp(only time hh:mm:ss)
                        })
    except KeyboardInterrupt:
        pass

    return JSONResponse(content={"status": "success", "attendances": recognized})

@atexit.register
def cleanup_camera():
    picam2.stop()