import cv2
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import csv
from datetime import datetime

# Configuration
EMBEDDINGS_PATH = "embeddings.pkl"
SIMILARITY_THRESHOLD = 0.6
MODEL_NAME = "ArcFace"
SESSION_DURATION = 60   # session length in seconds (1 min)
ATTENDANCE_THRESHOLD = 0.25  # 25%
OUTPUT_FILE = "attendance.csv"
MODEL_PATH_YUNET = 'face_detection_yunet_2023mar.onnx'  
DETECTOR = cv2.FaceDetectorYN.create(MODEL_PATH_YUNET, "", (320, 320))

# Load embeddings
def load_embeddings():
    with open(EMBEDDINGS_PATH, "rb") as f:
        return pickle.load(f)

# Find best match
def find_match(face_embedding, embeddings_db):
    best_match = "Unknown"
    best_score = 0
    
    for person, person_embeddings in embeddings_db.items():
        for stored_embedding in person_embeddings:
            similarity = cosine_similarity([face_embedding], [stored_embedding])[0][0]
            if similarity > best_score:
                best_score = similarity
                best_match = person
    
    return (best_match, best_score) if best_score >= SIMILARITY_THRESHOLD else ("Unknown", best_score)

# Save attendance to CSV
def save_attendance(attendance, session_start, session_end, session_length, session_name):
    filename = f"attendance_{session_name}.csv"
    file_exists = os.path.isfile(OUTPUT_FILE)
    
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:  # write header only first time
            writer.writerow(["Name", "Presence (s)", "Session Duration (s)", "Attendance (%)", "Status", "Start", "End"])
        
        for person, presence_time in attendance.items():
            percentage = presence_time / session_length
            status = "Present" if percentage >= ATTENDANCE_THRESHOLD else "Absent"
            writer.writerow([
                person, round(presence_time, 2), session_length,
                f"{percentage*100:.1f}%", status, 
                session_start.strftime("%Y-%m-%d %H:%M:%S"),
                session_end.strftime("%Y-%m-%d %H:%M:%S")
            ])



# Main recognition loop
def main(session_name,session_duration):

    # Load embeddings
    embeddings_db = load_embeddings()
    print(f"Loaded {len(embeddings_db)} people")
    
    # Session start & timers
    session_start = datetime.now()
    start_time = time.time()
    session_length = session_duration
    
    # Tracking presence
    presence_time = {person: 0 for person in embeddings_db.keys()}
    last_seen = {}  # track last detection time per person
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        try:
            DETECTOR.setInputSize((frame.shape[1], frame.shape[0]))

            # Detect faces
            _, faces = DETECTOR.detect(frame)  #[x, y, w, h, score, landmarks...]

            
            current_time = time.time()
            
            detected_people = set()
            
            if faces is not None:

                for face in faces:      
                    x, y, w, h = map(int, face[:4])
                    
                    # Extract face & embedding
                    face_img = frame[y:y+h, x:x+w]
                    
                    embedding = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
                    
                    name, confidence = find_match(embedding, embeddings_db)
                    detected_people.add(name)
                    
                    # Track presence time
                    if name != "Unknown":
                        if name not in last_seen:
                            last_seen[name] = current_time
                        else:
                            presence_time[name] += current_time - last_seen[name]
                            last_seen[name] = current_time
                    
                    # Draw results
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f}) {presence_time[name]}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Reset last_seen for people not detected in this frame
                for person in list(last_seen.keys()):
                    if person not in detected_people:
                        last_seen.pop(person)
                        
        except Exception as e:
            print(f"Error: {e}")
            
        cv2.imshow('Face Recognition Attendance', frame)
        
        # Stop session when time reached
        if current_time - start_time >= session_length:
            print("Session ended")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Session manually ended")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save attendance
    session_end = datetime.now()
    save_attendance(presence_time, session_start, session_end, session_length, session_name)


