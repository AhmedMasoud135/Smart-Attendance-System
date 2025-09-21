import cv2
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os
import time

# Configuration
EMBEDDINGS_PATH = "embeddings.pkl"
SIMILARITY_THRESHOLD = 0.6
MODEL_NAME = "ArcFace"
model_path = 'face_detection_yunet_2023mar.onnx'  
detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320))

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

# Main recognition loop
def main():
    # Load embeddings
    embeddings_db = load_embeddings()
    print(f"Loaded {len(embeddings_db)} people")
    
    # Start camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        
        try:
            # Detect faces

            # start = time.time()
            # result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # end = time.time()
            # print(f"⏱ Analysis time (DeepFace.analyze): {(end-start)*1000:.2f} ms")


            detector.setInputSize((frame.shape[1], frame.shape[0]))

            start = time.time()
            _, faces = detector.detect(frame)  #[x, y, w, h, score, landmarks...]
            end = time.time()
            print(f"⏱ Analysis time (cv2.FaceDetectorYN): {(end-start)*1000:.2f} ms")


           
            if faces is not None:
                for face in faces:
                    # Get coordinates
                    x, y, w, h = map(int, face[:4])
                    
                    # Extract face and get embedding
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Get embedding
                    embedding = DeepFace.represent(face_img, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
                    
                    # Find match
                    name, confidence = find_match(embedding, embeddings_db)
                    
                    # Draw results
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        except Exception as e:
            print(f"Error: {e}")
            
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

main()


# ⏱ Analysis time (cv2.FaceDetectorYN): 24.34 ms

# ⏱ Analysis time (DeepFace.analyze): 128.03 ms

# Big difference between DeepFace.analyze and cv2.FaceDetectorYN in analyzing time
# I need only where is the face so cv2.FaceDetectorYN is better 
