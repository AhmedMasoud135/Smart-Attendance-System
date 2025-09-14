# Smart Attendance System

## Description
This project implements a Smart Attendance System using face recognition. It leverages computer vision and deep learning techniques to identify individuals and track their presence during a session. The system records attendance, calculates presence percentages, and generates an attendance report.

## Features
- Real-time face detection and recognition.
- Tracks presence time for each recognized individual.
- Calculates attendance percentage based on session duration.
- Generates a CSV report with attendance details (Name, Presence, Session Duration, Attendance %, Status, Start Time, End Time).
- Configurable session duration and attendance threshold.

## Technologies Used
- Python
- OpenCV (`cv2`)
- DeepFace (for face recognition and embedding generation)
- NumPy
- scikit-learn (for cosine similarity)
- Pandas (implicitly for data handling, though not directly imported in `main.py` for core logic, it's common for data output)

## Setup and Installation
1.  **Clone the repository:** (Once uploaded to GitHub)
    ```bash
    git clone <your-repo-url>
    cd Smart-Attendance-System
    ```
2.  **Install dependencies:**
    ```bash
    pip install opencv-python deepface numpy scikit-learn pandas
    ```
3.  **Prepare Embeddings:** The system relies on `embeddings.pkl` which contains pre-computed face embeddings of known individuals. You would typically generate this using a separate script (e.g., `EncodeGenerator.py` based on the file list).

## Usage
To run the main attendance system, execute `Run.py`.

```bash
python Run.py
```

The system will use your webcam to detect faces. Attendance data will be saved to a CSV file (e.g., `attendance_your_session_name.csv`).

## File Structure
- `main.py`: The core script for real-time face recognition and attendance tracking.
- `EncodeGenerator.py`: Script to generate `embeddings.pkl` from a set of known faces.
- `GUI.py`: Script for a graphical user interface.
- `Run.py`: Entry point or utility script.
- `Student_Manage.py`: Script for managing student data.
- `test_face_detection.py`: Script for testing face detection functionality.
- `embeddings.pkl`: Pre-computed face embeddings database.
- `Images/`: Directory for storing images (e.g., for known faces).
- `README.md`: This file.





