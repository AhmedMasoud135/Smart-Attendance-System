import os
import shutil

import os
import cv2
import shutil
import time

# Path to store all student images
STUDENTS_DIR = "Smart Attendance System/Images"

def add_student(student_name, num_photos=20, delay=5):
    
    student_name = student_name.strip().replace(" ", "_")
    student_folder = os.path.join(STUDENTS_DIR, student_name)

    if not os.path.exists(STUDENTS_DIR):
        os.makedirs(STUDENTS_DIR)

    if os.path.exists(student_folder):
        print(f"⚠️ Student '{student_name}' already exists! Delete old folder if you want to recapture.")
        return

    os.makedirs(student_folder)

    cap = cv2.VideoCapture(0)  
    count = 0

    print("🎥 Camera opened. Press 'q' to quit, 'p' to capture manually.")

    last_capture_time = time.time() - delay  # allow immediate first capture

    while count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame from camera.")
            break

        # Show live preview with counter
        cv2.putText(frame, f"Photos: {count}/{num_photos}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Add Student", frame)

        key = cv2.waitKey(1) & 0xFF

        # Manual capture with 'p'
        if key == ord('p'):
            filename = os.path.join(student_folder, f"{student_name}_{count+1}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"📸 Captured (manual): {filename}")

        # Automatic capture every <delay> seconds
        elif time.time() - last_capture_time >= delay:
            filename = os.path.join(student_folder, f"{student_name}_{count+1}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            last_capture_time = time.time()
            print(f"📸 Captured (auto): {filename}")

        # Quit with 'q'
        elif key == ord('q'):
            print("🛑 Capture stopped manually.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Student '{student_name}' added successfully with {count} photos.")


def remove_student(student_name):
    student_name = student_name.strip().replace(" ", "_")
    student_folder = os.path.join(STUDENTS_DIR, student_name)

    if os.path.exists(student_folder):
        shutil.rmtree(student_folder)
        print(f"Student '{student_name}' removed successfully!")
    else:
        print(f"Student '{student_name}' not found.")

#add_student("John",5)
#remove_student("John")



