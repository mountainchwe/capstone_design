# bear's original code
import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
import math
import mediapipe as mp


print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [263, 387, 385, 362, 380, 373]


picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 1080)}))
#picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})) #okay
#picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (3280, 2464)})) #highest resolution but low fps
picam2.start()

# this has to be a whole number
# higher number -> bigger fps, bad performance
# lower number -> lower fps, better performance
cv_scaler = 4
SPOOF_TIMEOUT = 10
EAR_THRESHOLD = 0.2
YAW_THRESHOLD = 10
PITCH_THRESHOLD = 10

blink_total = 0
last_blink_time = time.time()
#spoofing_detected = False
spoofing_flags = []
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks_2d, w, h):
    # 6 key points for pose estimation
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

def detect_blink_and_head_movement(frame_rgb):
    global blink_total, last_blink_time

    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return blink_total, 0, 0

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame_rgb.shape
    landmarks_2d = [(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark]

    left_EAR = eye_aspect_ratio(landmarks_2d, LEFT_EYE_LANDMARKS)
    right_EAR = eye_aspect_ratio(landmarks_2d, RIGHT_EYE_LANDMARKS)
    avg_EAR = (left_EAR + right_EAR) / 2.0

    if avg_EAR < EAR_THRESHOLD:
        blink_total += 1
        last_blink_time = time.time()

    pitch, yaw, _ = get_head_pose(landmarks_2d, w, h)
    return blink_total, pitch, yaw

def process_frame(frame, frame_count):
    global face_locations, face_encodings, face_names, spoofing_flags
    
    spoofing_flags.clear()
    
    # Resize the frame using cv_scaler to increase performance (less pixels processed, less time spent)
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert the image from BGR to RGB colour space, the facial recognition library uses RGB, OpenCV uses BGR
    rgb_resized = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    if frame_count % 3 == 0:
        face_locations[:] = face_recognition.face_locations(rgb_resized)
        face_encodings[:] = face_recognition.face_encodings(rgb_resized, face_locations, model='small')
        face_names[:] = []
    
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
        
    blink_count, pitch, yaw = detect_blink_and_head_movement(rgb_resized)

    # Anti-spoofing check
    for name in face_names:
        if name == "Unknown":
            spoofing_flags.append("unknown")
        elif (time.time() - last_blink_time > SPOOF_TIMEOUT) or (abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD):
            spoofing_flags.append("spoofing")
        else:
            spoofing_flags.append("real")
    
    return frame

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name, status in zip(face_locations, face_names, spoofing_flags):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        if status == "unknown":
            box_color = (0, 0, 255)
            label = "Unknown"
        elif status == "spoofing":
            box_color = (0, 255, 255)  # Yellow
            label = f"{name}: Spoofing"
        else:
            box_color = (0, 255, 0)    # Green
            label = name
            
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (0, 0, 0), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

while True:
    # Capture a frame from camera
    frame = picam2.capture_array()
    
    # Process the frame with the function
    processed_frame = process_frame(frame, frame_count)
    
    # Get the text and boxes to be drawn based on the processed frame
    display_frame = draw_results(processed_frame)
    
    # Calculate and update FPS
    current_fps = calculate_fps()
    
    # Attach FPS counter to the text and boxes
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display everything over the video feed.
    cv2.imshow('Video', display_frame)
    
    # Break the loop and stop the script if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# By breaking the loop we run this code here which closes everything
cv2.destroyAllWindows()
picam2.stop()