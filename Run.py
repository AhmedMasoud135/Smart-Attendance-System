from main import main
import time
from datetime import datetime

# Run scheduled session
def scheduler(sessions):
    
    print(f"Loaded {len(sessions)} scheduled sessions")

    for session_name, info in sessions.items():
        start_str = info["start"]
        duration = info["duration_minutes"] * 60
        start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")

        # Append date to session name
        date_suffix = start_time.strftime("%Y-%m-%d")
        session_name_with_date = f"{session_name}_{date_suffix}"

        print(f"Waiting for session '{session_name_with_date}' at {start_str}...")
        while datetime.now() < start_time:
            time.sleep(5)  # check every 5 seconds
        
        print(f"Starting session '{session_name_with_date}'...")
        main(session_name_with_date, duration)


sessions = {
        "Session1": {"start": "2025-09-02 06:17:00", "duration_minutes": 3},
        "Session2": {"start": "2025-09-02 14:00:00", "duration_minutes": 1},
        "Session3": {"start": "2025-09-03 14:00:00", "duration_minutes": 1}

    }

scheduler(sessions)