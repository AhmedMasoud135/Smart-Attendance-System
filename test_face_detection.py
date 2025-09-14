import cv2
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os

# Configuration
EMBEDDINGS_PATH = "embeddings.pkl"
SIMILARITY_THRESHOLD = 0.6
MODEL_NAME = "ArcFace"

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
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if not isinstance(result, list):
                result = [result]
            
            for face in result:
                # Get coordinates
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
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
