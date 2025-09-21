import os
import cv2
import pickle
from deepface import DeepFace
import imgaug.augmenters as iaa
import numpy as np

def manage_embeddings(db_path="Smart Attendance System/Images", N_AUG=5, emb_path="embeddings.pkl"):
    """
    This function manages face embeddings:
    - If embeddings.pkl does not exist, it creates embeddings for all students.
    - If embeddings.pkl exists, it only updates by adding embeddings for new students.
    """

    # Augmentation pipeline
    augmenter = iaa.Sequential([
        iaa.Fliplr(0.5),                     # horizontal flip
        iaa.GaussianBlur(sigma=(0, 1)),      # blur
        iaa.Multiply((0.8, 1.2)),            # brightness
        iaa.LinearContrast((0.8, 1.2)),      # contrast
        iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),  # noise
        iaa.Affine(rotate=(-15, 15), shear=(-10, 10), scale=(0.9, 1.1))  # rotation, shear, zoom
    ])

    # Load existing embeddings if available
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as f:
            embeddings = pickle.load(f)
        print("📂 Loaded existing embeddings.")
    else:
        embeddings = {}
        print("🆕 Starting fresh embeddings database.")

    def save_temp_image(image, temp_path="temp_aug.jpg"):
        """Helper function: save augmented image temporarily to disk"""
        cv2.imwrite(temp_path, image)
        return temp_path

    # Loop through each student folder 
    for person in os.listdir(db_path):
        person_path = os.path.join(db_path, person)
        if not os.path.isdir(person_path):
            continue

        # Skip if already in embeddings
        if person in embeddings:
            print(f"⏩ Skipping {person} (already encoded).")
            continue

        print(f"\n🔄 Processing new person: {person}")
        embeddings[person] = []

        # Loop through each image of the student 
        for img in os.listdir(person_path):
            if not img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            img_path = os.path.join(person_path, img)

            try:
                # Generate embedding for the original image
                rep = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)
                embeddings[person].append(rep[0]["embedding"])
                print(f"  ✅ Original processed: {img}")

                # Augmentation and embedding 
                face_img = cv2.imread(img_path)
                if face_img is None:
                    print(f"  ⚠️ Could not read image: {img}")
                    continue

                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                for i in range(N_AUG):
                    # Apply augmentation
                    aug_img_rgb = augmenter(image=face_img_rgb)
                    aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)


                    # Generate embedding for augmented image
                    aug_rep = DeepFace.represent(img_path=aug_img_bgr, model_name="ArcFace", enforce_detection=False)
                    embeddings[person].append(aug_rep[0]["embedding"])

                
                    print(f"  ✅ Augmentation {i+1}/{N_AUG} processed")

            except Exception as e:
                print(f"  ❌ Skipping {img_path}: {e}")

        print(f"📊 Total embeddings for {person}: {len(embeddings[person])}")

    # Remove embeddings of students no longer in folder
    current_students = set(os.listdir(db_path))
    to_remove = [name for name in embeddings if name not in current_students]

    for name in to_remove:
        del embeddings[name]
        print(f"🗑️ Removed old student embeddings: {name}")

    # Save updated embeddings 
    with open(emb_path, "wb") as f:
        pickle.dump(embeddings, f)

    print("\n✅ Embeddings updated & saved.")
    total_embeddings = sum(len(v) for v in embeddings.values())
    print(f"📈 Final Summary: {len(embeddings)} people, {total_embeddings} embeddings total.")

manage_embeddings(N_AUG=1)