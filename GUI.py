import os
import sys
import json
import csv
import time
import shutil
import threading
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# theme 
try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import *
    THEME = True
except Exception:
    THEME = False

# imaging / camera
try:
    import cv2
    from PIL import Image, ImageTk
    CV2 = True
except Exception:
    CV2 = False

# face libs 
try:
    import pickle
    from deepface import DeepFace
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    FACE = True
except Exception:
    FACE = False

# external project modules
try:
    from Student_Manage import add_student as ext_add_student, remove_student as ext_remove_student
    EXT_STUDENT = True
except Exception:
    EXT_STUDENT = False
    ext_add_student = None
    ext_remove_student = None

try:
    from EncodeGenerator import manage_embeddings as ext_manage_embeddings
    EXT_EMBEDDINGS = True
except Exception:
    EXT_EMBEDDINGS = False
    ext_manage_embeddings = None

# CONFIG 
SESSIONS_FILE = "sessions.json"
EMBEDDINGS_FILE = "embeddings.pkl"
STUDENTS_DIR = "Smart Attendance System/Images"
ATTENDANCE_PREFIX = "attendance_"
SIMILARITY_THRESHOLD = 0.60
ATTENDANCE_THRESHOLD = 0.25  # 25%

# SHARED STATE 
STATE = {
    "root": None,
    "video_label": None,            # live attendance preview
    "student_video_label": None,    # student capture preview
    "log_text": None,
    "sessions_tree": None,
    "live_tree": None,
    "attendance_thread": None,
    "scheduler_thread": None,
    "attendance_running": False,
    "attendance_stop": False,
    "session_name": None,
    "session_start_dt": None,
    "session_duration_s": 0,
    "presence_time": {},
    "last_seen": {},
    "embeddings_db": None,
    "student_capture_thread": None,
    "student_capture_running": False,
    "student_current_frame": None,
    "student_capture_count": 0,
    "student_capture_target": 0,
    "student_capture_delay": 5,
    "student_capture_name": "",
    "student_cap": None,
}

# UTIL
def ensure_dirs():
    """Ensure the students images directory exists."""
    os.makedirs(STUDENTS_DIR, exist_ok=True)

def log(msg):
    """
    Thread-safe GUI log function.
    Writes message to the log text widget if available, otherwise to stdout.
    """
    txt = STATE.get("log_text")
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{timestamp}  {msg}\n"
    if txt:
        def _append():
            try:
                txt.configure(state="normal")
                txt.insert("end", line)
                txt.see("end")
                txt.configure(state="disabled")
            except Exception:
                # fallback to printing if GUI isn't available
                print(line, end='')
        try:
            txt.after(0, _append)
        except Exception:
            print(line, end='')
    else:
        print(line, end='')

def load_sessions():
    """
    Load sessions dictionary from SESSIONS_FILE.
    Returns an empty dict if file does not exist or is invalid.
    The structure is:
    sessions = {
        "SessionName": {"start": "YYYY-MM-DD HH:MM:SS", "duration_minutes": 30},
        ...
    }
    """
    if not os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "w") as f:
                json.dump({}, f, indent=2)
        except Exception as e:
            log(f"❌ Failed to create {SESSIONS_FILE}: {e}")
        return {}
    try:
        with open(SESSIONS_FILE, "r") as f:
            data = json.load(f)
            # ensure keys are strings and values contain expected keys
            if not isinstance(data, dict):
                log("⚠️ sessions.json content invalid, resetting to {}.")
                return {}
            return data
    except Exception as e:
        log(f"❌ sessions.json read error: {e}")
        return {}

def save_sessions(sessions):
    """Write sessions dict to SESSIONS_FILE (atomically)."""
    try:
        tmp = SESSIONS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sessions, f, indent=2)
        os.replace(tmp, SESSIONS_FILE)
        log(f"💾 sessions saved ({len(sessions)} sessions).")
    except Exception as e:
        log(f"❌ Failed to save sessions: {e}")

def load_embeddings_pickle():
    """
    Load embeddings dictionary from EMBEDDINGS_FILE.
    Return empty dict if not available. This function logs status.
    """
    if not FACE:
        log("⚠️ DeepFace/sklearn not installed; recognition disabled.")
        return {}
    if not os.path.exists(EMBEDDINGS_FILE):
        log(f"⚠️ Embeddings file '{EMBEDDINGS_FILE}' not found.")
        return {}
    try:
        with open(EMBEDDINGS_FILE, "rb") as f:
            emb = pickle.load(f)
        log(f"Loaded embeddings for {len(emb)} people.")
        return emb
    except Exception as e:
        log(f"❌ Failed to load embeddings: {e}")
        return {}

def find_match(embedding, embeddings_db):
    """
    Find best match for an embedding in embeddings_db using cosine similarity.
    Returns (name, score) or ("Unknown", score).
    """
    if not embeddings_db:
        return ("Unknown", 0.0)
    best = -1.0
    person = "Unknown"
    for name, emb_list in embeddings_db.items():
        for se in emb_list:
            try:
                s = cosine_similarity([embedding], [se])[0][0]
            except Exception:
                s = 0.0
            if s > best:
                best = s
                person = name
    if best >= SIMILARITY_THRESHOLD:
        return (person, best)
    return ("Unknown", best)

# SESSIONS UI
def refresh_sessions_table():
    """
    Refresh the Sessions table widget with data from sessions.json.
    This reads the file each time so external edits are also reflected.
    """
    tree = STATE.get("sessions_tree")
    if tree is None:
        return
    for r in tree.get_children():
        tree.delete(r)
    sessions = load_sessions()
    now = datetime.now()
    # Sort by start time if parseable, else by name
    def sort_key(item):
        name, info = item
        start = info.get("start") if isinstance(info, dict) else ""
        try:
            dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            return (dt, name)
        except Exception:
            return (datetime.max, name)
    for name, info in sorted(sessions.items(), key=sort_key):
        start = info.get("start", "")
        dur = info.get("duration_minutes", "")
        # Determine status
        try:
            dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
            if now < dt:
                status = "Scheduled"
            elif now < dt + timedelta(minutes=int(dur)):
                status = "Active"
            else:
                status = "Completed"
        except Exception:
            status = "Invalid"
        tree.insert("", "end", iid=name, values=(name, start, dur, status))

def add_or_update_session_from_form(name_var, date_var, time_var, dur_var, editing_original=None):
    """
    Called by the Add/Update button in the Sessions form.
    It validates input, merges into existing sessions and saves.
    """
    name = name_var.get().strip()
    date = date_var.get().strip()
    t = time_var.get().strip()
    dur = dur_var.get()
    if not name or not date or not t:
        messagebox.showerror("Input", "Please fill name, date and time.")
        return
    dt_str = f"{date} {t}"
    try:
        # validate datetime format
        datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        messagebox.showerror("Format", "Date/time format must be YYYY-MM-DD and HH:MM:SS")
        return

    # Load current sessions, update/merge (fix for overwrite)
    sessions = load_sessions()
    # If editing original name and changed name, remove old key
    if editing_original and editing_original != name:
        if editing_original in sessions:
            sessions.pop(editing_original)
    sessions[name] = {"start": dt_str, "duration_minutes": int(dur)}
    save_sessions(sessions)  # write merged dict back
    log(f"➕ Session '{name}' saved for {dt_str} ({dur} min)")
    refresh_sessions_table()

def on_edit_selected_fill(form_name_var, form_date_var, form_time_var, form_dur_var, current_edit_var):
    """
    Fill the session form with the selected session's data so user can edit.
    """
    tree = STATE.get("sessions_tree")
    sel = tree.selection()
    if not sel:
        messagebox.showinfo("Select", "Select a session to edit.")
        return
    iid = sel[0]
    vals = tree.item(iid)['values']
    if not vals:
        return
    name, start_str, dur = vals[0], vals[1], vals[2]
    try:
        dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        form_name_var.set(name)
        form_date_var.set(dt.strftime("%Y-%m-%d"))
        form_time_var.set(dt.strftime("%H:%M:%S"))
        form_dur_var.set(int(dur))
        current_edit_var.set(name)  # track original name
        log(f"✏️ Editing session '{name}'")
    except Exception:
        messagebox.showerror("Error", "Invalid session start format.")

def delete_selected_session():
    """
    Remove the selected session from sessions.json
    """
    tree = STATE.get("sessions_tree")
    sel = tree.selection()
    if not sel:
        messagebox.showinfo("Select", "Select a session to delete.")
        return
    name = sel[0]
    sessions = load_sessions()
    if name in sessions:
        if not messagebox.askyesno("Confirm", f"Delete session '{name}'?"):
            return
        sessions.pop(name, None)
        save_sessions(sessions)
        log(f"🗑️ Deleted session '{name}'")
        refresh_sessions_table()
    else:
        messagebox.showerror("Error", f"Session '{name}' not found")

# ATTENDANCE 
def update_live_table():
    """
    Update the live attendance treeview using the current STATE presence_time and last_seen.
    """
    tree = STATE.get("live_tree")
    if not tree:
        return
    existing = {tree.item(iid)['values'][0]: iid for iid in tree.get_children()}
    names = set(list(STATE.get("presence_time", {}).keys()) + list(STATE.get("last_seen", {}).keys()))
    for name in names:
        pres = STATE["presence_time"].get(name, 0.0)
        last_ts = STATE["last_seen"].get(name)
        last = datetime.fromtimestamp(last_ts).strftime("%H:%M:%S") if last_ts else "-"
        dur = STATE.get("session_duration_s", 0)
        status = "Present" if (dur>0 and pres/dur >= ATTENDANCE_THRESHOLD) else "Absent"
        if name in existing:
            iid = existing[name]
            tree.item(iid, values=(name, last, round(pres,1), status))
        else:
            tree.insert("", "end", values=(name, last, round(pres,1), status))

def save_attendance_csv(session_name):
    """
    Save attendance for the given session_name to a dedicated CSV file.
    Only writes presence_time from STATE to ensure one file per session.
    """
    fname = f"{ATTENDANCE_PREFIX}{session_name}.csv"
    try:
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Name", "Presence (s)", "Session Duration (s)", "Attendance (%)", "Status", "Start", "End"])
            for name, pres in STATE["presence_time"].items():
                dur = STATE.get("session_duration_s", 0)
                pct = (pres/dur) if dur>0 else 0.0
                status = "Present" if pct >= ATTENDANCE_THRESHOLD else "Absent"
                start = STATE.get("session_start_dt", datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
                end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                w.writerow([name, round(pres,2), dur, f"{pct*100:.1f}%", status, start, end])
        log(f"📥 Attendance saved to '{fname}'")
    except Exception as e:
        log(f"❌ Failed saving CSV: {e}")

def attendance_thread_func(session_display_name, duration_seconds):
    """
    Main attendance recognition loop. Runs in background thread.
    Updates STATE presence_time and last_seen; renders camera frames into GUI.
    """
    if not CV2 or not FACE:
        log("❌ Attendance requires OpenCV + DeepFace + sklearn + numpy. Disabled.")
        return

    STATE["embeddings_db"] = load_embeddings_pickle()
    # initialize presence_time for known people to ensure appearing in the table
    STATE["presence_time"] = {k:0.0 for k in (STATE["embeddings_db"].keys() if STATE["embeddings_db"] else [])}
    STATE["last_seen"] = {}
    STATE["live_people"] = {}
    STATE["session_name"] = session_display_name
    STATE["session_start_dt"] = datetime.now()
    STATE["session_duration_s"] = duration_seconds

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log("❌ Camera not accessible.")
        STATE["attendance_running"] = False
        return

    STATE["attendance_running"] = True
    STATE["attendance_stop"] = False
    log(f"▶ Session '{session_display_name}' started ({duration_seconds}s)")

    # compute end timestamp (time.time() used for accurate wall-clock duration)
    end_ts = time.time() + duration_seconds
    # show frames and detect
    while time.time() < end_ts and not STATE["attendance_stop"]:
        ret, frame = cap.read()
        if not ret:
            log("❌ Failed to read camera frame.")
            break
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        # detect faces & represent
        try:
            # Use DeepFace.analyze to get face regions (we only use region)
            results = DeepFace.analyze(display, actions=['emotion'], enforce_detection=False)
            if not isinstance(results, list):
                results = [results]
            detected = set()
            now_ts = time.time()
            for face in results:
                region = face.get("region", {})
                x = int(region.get("x", 0)); y = int(region.get("y", 0))
                w = int(region.get("w", 0)); h = int(region.get("h", 0))
                # guard invalid boxes
                if w<=0 or h<=0:
                    continue
                y2, x2 = y+h, x+w
                if y<0 or x<0 or y2>display.shape[0] or x2>display.shape[1]:
                    continue
                face_img = display[y:y+h, x:x+w].copy()
                tmpf = "temp_gui_face.jpg"
                cv2.imwrite(tmpf, face_img)
                rep = DeepFace.represent(img_path=tmpf, model_name="ArcFace", enforce_detection=False)
                if os.path.exists(tmpf):
                    try: os.remove(tmpf)
                    except: pass
                if rep and len(rep)>0:
                    emb = rep[0]["embedding"]
                    name, conf = find_match(emb, STATE["embeddings_db"])
                else:
                    name, conf = ("Unknown", 0.0)
                detected.add(name)
                # accumulate presence time
                if name != "Unknown":
                    if name not in STATE["last_seen"]:
                        STATE["last_seen"][name] = now_ts
                    else:
                        STATE["presence_time"][name] += now_ts - STATE["last_seen"][name]
                        STATE["last_seen"][name] = now_ts
                color = (0,255,0) if name!="Unknown" else (0,0,255)
                cv2.rectangle(display, (x,y), (x+w,y+h), color, 2)
                cv2.putText(display, f"{name} {conf:.2f}", (x, max(0,y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # remove people not detected in this frame (so last_seen only contains currently-detected people)
            for p in list(STATE["last_seen"].keys()):
                if p not in detected:
                    STATE["last_seen"].pop(p, None)
        except Exception as e:
            # Non-fatal: log and continue
            log(f"⚠️ Recognition error: {e}")

        # render display into GUI label
        try:
            img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            # Lazy import for Image if not already imported
            from PIL import Image as PILImage
            img = PILImage.fromarray(img).resize((720,480))
            imgtk = ImageTk.PhotoImage(image=img)
            lbl = STATE.get("video_label")
            if lbl:
                def _upd():
                    lbl.imgtk = imgtk
                    lbl.configure(image=imgtk)
                lbl.after(0, _upd)
        except Exception as e:
            log(f"⚠️ GUI render error: {e}")

        # update live table periodically (once in a short interval)
        if int(time.time()*2) % 2 == 0:
            try:
                STATE.get("root").after(0, update_live_table)
            except Exception:
                pass

        # small sleep to avoid CPU spin
        time.sleep(0.05)

    # done: cleanup camera and save attendance CSV
    try:
        cap.release()
    except:
        pass
    STATE["attendance_running"] = False
    save_attendance_csv(session_display_name)
    log(f"✔ Session '{session_display_name}' finished and saved.")
    # clear preview in GUI
    lbl = STATE.get("video_label")
    if lbl:
        lbl.after(0, lambda: lbl.configure(image='', text="No live feed"))
    # final update of live table
    try:
        STATE.get("root").after(0, update_live_table)
    except Exception:
        pass

def start_session_thread_from_selection(name, info):
    """
    Start a session when user clicks Start Selected Now or scheduler triggers it.
    FIX 2 applied here: ensure we call the wrapper with (name, info), not with mismatched args.
    """
    if STATE["attendance_running"]:
        messagebox.showinfo("Running", "Another session is already running.")
        return
    try:
        # Use wrapper that accepts (base_name, info)
        t = threading.Thread(target=attendance_thread_func_wrapper, args=(name, info), daemon=True)
        STATE["attendance_thread"] = t
        t.start()
    except Exception as e:
        messagebox.showerror("Error", f"Invalid session data: {e}")
        log(f"❌ Failed to start session thread: {e}")

def attendance_thread_func_wrapper(base_name, info):
    """
    Wrapper that computes start_dt and duration_seconds, then calls attendance_thread_func.
    This centralizes parsing and prevents type-mismatch errors.
    """
    try:
        start_dt = datetime.strptime(info["start"], "%Y-%m-%d %H:%M:%S")
        dur_s = int(info["duration_minutes"]) * 60
    except Exception as e:
        log(f"Invalid session data for '{base_name}': {e}")
        return
    session_display = f"{base_name}_{start_dt.strftime('%Y-%m-%d')}"
    attendance_thread_func(session_display, dur_s)

def stop_current_session():
    """Signal current attendance thread to stop (graceful)."""
    if STATE["attendance_running"]:
        STATE["attendance_stop"] = True
        log("🛑 Stop requested for current session (will stop between frames).")
    else:
        log("ℹ️ No session running.")

# STUDENT CAPTURE (embedded) 
def start_student_capture(name, photos_count, delay_s):
    """
    Embedded student capture in Students tab.
    - Shows preview in the student_video_label
    - Captures automatically every `delay_s` seconds until `photos_count` reached
    - Also supports manual capture by pressing 'p' while the GUI window has focus
    - When target reached, capture stops and camera is released, and preview cleared (FIX 3)
    """
    if not CV2:
        messagebox.showerror("Missing", "OpenCV/Pillow required for capture.")
        return
    name = name.strip().replace(" ", "_")
    if not name:
        messagebox.showerror("Input", "Enter student name.")
        return

    folder = os.path.join(STUDENTS_DIR, name)
    os.makedirs(folder, exist_ok=True)

    # initialize STATE flags
    STATE["student_capture_running"] = True
    # count existing images so we continue numbering
    STATE["student_capture_count"] = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png'))])
    STATE["student_capture_target"] = int(photos_count)
    STATE["student_capture_delay"] = int(delay_s)
    STATE["student_capture_name"] = name
    STATE["student_capture_last_auto"] = time.time()
    STATE["student_cap"] = None

    log(f"📸 Student capture started for '{name}'. Note: press 'p' to capture manually. Will auto-stop at {photos_count} photos.")

    # bind 'p' key for manual capture (only while running)
    def key_handler(event):
        if STATE["student_capture_running"] and hasattr(event, "keysym") and event.keysym.lower() == 'p':
            manual_capture_student()
    root = STATE["root"]
    if root:
        root.bind_all('<Key>', key_handler)

    def worker():
        cap = cv2.VideoCapture(0)
        STATE["student_cap"] = cap
        if not cap.isOpened():
            log("❌ Camera not accessible for student capture.")
            STATE["student_capture_running"] = False
            try:
                if root:
                    root.unbind_all('<Key>')
            except Exception:
                pass
            return
        lbl = STATE.get("student_video_label")
        # local last auto capture time
        last_auto = STATE.get("student_capture_last_auto", time.time())
        try:
            while STATE["student_capture_running"]:
                ret, frame = cap.read()
                if not ret:
                    log("❌ Failed to read frame during student capture.")
                    break
                frame = cv2.flip(frame, 1)
                # overlay counter
                c = STATE["student_capture_count"]
                target = STATE["student_capture_target"]
                cv2.putText(frame, f"{c}/{target}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

                # update preview in GUI
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Delay import to keep startup light if PIL not available until needed
                    from PIL import Image as PILImage
                    img_p = PILImage.fromarray(img).resize((480,360))
                    imgtk = ImageTk.PhotoImage(image=img_p)
                    def _upd():
                        if lbl:
                            lbl.imgtk = imgtk
                            lbl.configure(image=imgtk)
                            STATE["student_current_frame"] = frame.copy()
                    if lbl:
                        lbl.after(0, _upd)
                except Exception:
                    pass

                # Automatic capture check
                now = time.time()
                if STATE["student_capture_count"] < STATE["student_capture_target"] and (now - last_auto >= STATE["student_capture_delay"]):
                    try:
                        cnt = STATE["student_capture_count"] + 1
                        path = os.path.join(folder, f"{name}_{cnt}.jpg")
                        cv2.imwrite(path, frame)
                        STATE["student_capture_count"] = cnt
                        last_auto = now
                        STATE["student_capture_last_auto"] = now
                        log(f"✅ Auto-captured: {path}")
                    except Exception as e:
                        log(f"❌ Auto-capture failed: {e}")

                # Check completion
                if STATE["student_capture_count"] >= STATE["student_capture_target"]:
                    log(f"🎉 Student '{name}' capture complete ({STATE['student_capture_count']} photos).")
                    # stop capturing; break to clean up
                    STATE["student_capture_running"] = False
                    break

                time.sleep(0.03)
        finally:
            # CLEANUP: ensure we release camera and destroy windows (FIX 3)
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                # close OpenCV windows (if any were opened)
                cv2.destroyAllWindows()
            except Exception:
                pass
            STATE["student_capture_running"] = False
            STATE["student_current_frame"] = None
            # clear preview after done
            try:
                if lbl:
                    lbl.after(0, lambda: lbl.configure(image='', text="Student camera preview"))
            except Exception:
                pass
            # Unbind key handler
            try:
                if root:
                    root.unbind_all('<Key>')
            except Exception:
                pass

    th = threading.Thread(target=worker, daemon=True)
    STATE["student_capture_thread"] = th
    th.start()

def manual_capture_student():
    """
    Capture the current student frame to disk when 'p' pressed or manual capture called.
    """
    if not STATE.get("student_capture_running"):
        log("⚠️ Student capture not running.")
        return
    frame = STATE.get("student_current_frame")
    if frame is None:
        log("⚠️ No frame available to capture.")
        return
    name = STATE.get("student_capture_name")
    folder = os.path.join(STUDENTS_DIR, name)
    cnt = STATE["student_capture_count"] + 1
    path = os.path.join(folder, f"{name}_{cnt}.jpg")
    try:
        cv2.imwrite(path, frame)
        STATE["student_capture_count"] = cnt
        log(f"📸 Manual capture saved: {path}")
        if STATE["student_capture_count"] >= STATE["student_capture_target"]:
            log(f"🎉 Student '{name}' capture complete ({cnt} photos).")
            STATE["student_capture_running"] = False
    except Exception as e:
        log(f"❌ Failed to save manual capture: {e}")

def stop_student_capture():
    """Allow user to stop student capture manually (will cleanup in worker)."""
    if STATE.get("student_capture_running"):
        STATE["student_capture_running"] = False
        log("🛑 Student capture stopped manually.")
    else:
        log("ℹ️ Student capture is not running.")

# SCHEDULER 
def scheduler_worker():
    """
    Background scheduler that checks sessions.json and launches sessions that
    are currently within their start->end window. Runs forever in a daemon thread.
    """
    log("▶ Scheduler started (background).")
    while True:
        try:
            sessions = load_sessions()
            now = datetime.now()
            for name, info in sessions.items():
                try:
                    start_dt = datetime.strptime(info["start"], "%Y-%m-%d %H:%M:%S")
                    dur_s = int(info["duration_minutes"]) * 60
                except Exception:
                    # invalid session entry – skip
                    continue
                # if current time is within session window and not already running, launch
                if start_dt <= now < start_dt + timedelta(seconds=dur_s):
                    if not STATE["attendance_running"]:
                        session_display = f"{name}_{start_dt.strftime('%Y-%m-%d')}"
                        log(f"Scheduler launching session '{session_display}'")
                        t = threading.Thread(target=attendance_thread_func_wrapper, args=(name, info), daemon=True)
                        STATE["attendance_thread"] = t
                        t.start()
                        # wait the session duration plus a small buffer to avoid relaunching same session
                        time.sleep(dur_s + 2)
            time.sleep(3)
        except Exception as e:
            log(f"Scheduler error: {e}")
            time.sleep(5)

# GUI BUILD 
def build_gui():
    """
    Construct the main GUI: Notebook tabs for Sessions, Live Attendance, Students.
    Wire up buttons and widgets to the functions above.
    """
    ensure_dirs()
    # root window
    if THEME:
        root = tb.Window(themename="darkly")
    else:
        root = tk.Tk()
        root.title("Smart Attendance System")
    STATE["root"] = root
    root.geometry("1200x820")

    # Title
    title = ttk.Label(root, text="📚 Smart Attendance System", font=("Segoe UI", 16, "bold"))
    title.pack(anchor="nw", padx=12, pady=(6,0))

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=12, pady=8)

    # Sessions tab
    tab_sess = ttk.Frame(nb)
    nb.add(tab_sess, text="📅 Sessions")
    # form
    form = ttk.LabelFrame(tab_sess, text="Add / Update Session", padding=8)
    form.pack(fill="x", padx=8, pady=8)
    sess_name_var = tk.StringVar()
    sess_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
    sess_time_var = tk.StringVar(value=datetime.now().strftime("%H:%M:%S"))
    sess_dur_var = tk.IntVar(value=30)
    editing_original = tk.StringVar(value="")

    ttk.Label(form, text="Name:").grid(row=0, column=0, sticky="w")
    ttk.Entry(form, textvariable=sess_name_var, width=30).grid(row=0, column=1, padx=6, pady=4)
    ttk.Label(form, text="Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w")
    ttk.Entry(form, textvariable=sess_date_var, width=20).grid(row=1, column=1, sticky="w", padx=6)
    ttk.Label(form, text="Time (HH:MM:SS):").grid(row=2, column=0, sticky="w")
    ttk.Entry(form, textvariable=sess_time_var, width=20).grid(row=2, column=1, sticky="w", padx=6)
    ttk.Label(form, text="Duration (minutes):").grid(row=3, column=0, sticky="w")
    ttk.Spinbox(form, from_=1, to=480, textvariable=sess_dur_var, width=8).grid(row=3, column=1, sticky="w", padx=6)

    ttk.Button(form, text="➕ Add / Update", command=lambda: add_or_update_session_from_form(sess_name_var, sess_date_var, sess_time_var, sess_dur_var, editing_original.get())).grid(row=4, column=0, columnspan=2, pady=8)
    ttk.Button(form, text="✏️ Edit Selected", command=lambda: on_edit_selected_fill(sess_name_var, sess_date_var, sess_time_var, sess_dur_var, editing_original)).grid(row=5, column=0, pady=4)
    ttk.Button(form, text="🗑 Delete Selected", command=delete_selected_session).grid(row=5, column=1, pady=4)

    # sessions table
    tbl_frame = ttk.LabelFrame(tab_sess, text="Scheduled Sessions", padding=6)
    tbl_frame.pack(fill="both", expand=True, padx=8, pady=8)
    cols = ("name", "start", "dur", "status")
    tree = ttk.Treeview(tbl_frame, columns=cols, show="headings", height=10)
    for c,txt,w in zip(cols, ("Name","Start","Duration(min)","Status"), (220,220,120,120)):
        tree.heading(c, text=txt); tree.column(c, width=w, anchor="center")
    tree.pack(fill="both", expand=True)
    STATE["sessions_tree"] = tree
    refresh_sessions_table()

    # Live Attendance tab
    tab_live = ttk.Frame(nb)
    nb.add(tab_live, text="📋 Live Attendance")
    topc = ttk.Frame(tab_live); topc.pack(fill="x", padx=8, pady=6)
    ttk.Button(topc, text="▶ Start Selected Now", command=lambda: start_selected_now_from_sessions()).pack(side="left", padx=6)
    ttk.Button(topc, text="🛑 Stop Session", command=stop_current_session).pack(side="left", padx=6)
    ttk.Button(topc, text="📥 Download Current Session CSV", command=download_current_session_csv).pack(side="left", padx=6)
    # preview + live table
    center = ttk.Frame(tab_live); center.pack(fill="both", expand=True, padx=8, pady=6)
    left = ttk.LabelFrame(center, text="Camera Preview", padding=6); left.pack(side="left", fill="both", expand=True, padx=(0,6))
    video_lbl = ttk.Label(left, text="No live feed"); video_lbl.pack(fill="both", expand=True)
    STATE["video_label"] = video_lbl
    right = ttk.LabelFrame(center, text="Live Recognized People", padding=6); right.pack(side="left", fill="both", expand=True)
    lcols = ("name","last_seen","presence","status")
    lt = ttk.Treeview(right, columns=lcols, show="headings", height=12)
    for c,txt in zip(lcols, ("Name","Last Seen","Presence(s)","Status")):
        lt.heading(c, text=txt); lt.column(c, anchor="center", width=140)
    lt.pack(fill="both", expand=True)
    STATE["live_tree"] = lt

    # Students tab
    tab_stud = ttk.Frame(nb); nb.add(tab_stud, text="👥 Students")
    topf = ttk.LabelFrame(tab_stud, text="Add Student (embedded capture)", padding=8); topf.pack(fill="x", padx=8, pady=6)
    s_name_var = tk.StringVar()
    s_photos_var = tk.IntVar(value=20)
    s_delay_var = tk.IntVar(value=5)
    ttk.Label(topf, text="Student name:").grid(row=0, column=0, sticky="w")
    ttk.Entry(topf, textvariable=s_name_var, width=25).grid(row=0, column=1, padx=6, pady=4)
    ttk.Label(topf, text="Photos:").grid(row=1, column=0, sticky="w")
    ttk.Spinbox(topf, from_=1, to=200, textvariable=s_photos_var, width=6).grid(row=1, column=1, sticky="w")
    ttk.Label(topf, text="Delay (s):").grid(row=2, column=0, sticky="w")
    ttk.Spinbox(topf, from_=0, to=60, textvariable=s_delay_var, width=6).grid(row=2, column=1, sticky="w")
    ttk.Button(topf, text="➕ Add Student (start capture)", command=lambda: start_student_capture(s_name_var.get(), s_photos_var.get(), s_delay_var.get())).grid(row=3, column=0, columnspan=2, pady=6)
    ttk.Label(topf, text="Note: Press 'p' to capture manually during capture. Camera will stop automatically when target photos captured.").grid(row=4, column=0, columnspan=2, sticky="w")
    # student preview below
    sframe = ttk.LabelFrame(tab_stud, text="Preview", padding=6); sframe.pack(fill="both", expand=True, padx=8, pady=6)
    svid = ttk.Label(sframe, text="Student camera preview")
    svid.pack(fill="both", expand=True)
    STATE["student_video_label"] = svid
    # remove / update embeddings
    rightf = ttk.LabelFrame(tab_stud, text="Manage", padding=6); rightf.pack(fill="x", padx=8, pady=6)
    students = get_student_list()
    scombo_var = tk.StringVar()
    scombo = ttk.Combobox(rightf, values=students, textvariable=scombo_var, width=30)
    scombo.pack(side="left", padx=6, pady=4)
    ttk.Button(rightf, text="❌ Remove Selected", command=lambda: remove_student_folder(scombo_var.get(), scombo)).pack(side="left", padx=6)
    ttk.Button(rightf, text="🔄 Update embeddings", command=start_update_embeddings).pack(side="left", padx=6)
    ttk.Button(rightf, text="🔄 Refresh list", command=lambda: refresh_student_combobox(scombo)).pack(side="left", padx=6)

    # console log
    logf = ttk.LabelFrame(root, text="Console Log", padding=6); logf.pack(fill="both", padx=12, pady=(0,10))
    logtxt = tk.Text(logf, height=8, state="disabled")
    logtxt.pack(fill="both", expand=True)
    STATE["log_text"] = logtxt

    # start scheduler thread if not already started
    if not STATE.get("scheduler_thread"):
        th = threading.Thread(target=scheduler_worker, daemon=True)
        STATE["scheduler_thread"] = th
        th.start()

    return root

# Helper UI functions (for button callbacks) 
def start_selected_now_from_sessions():
    """
    Called by 'Start Selected Now' button on Live tab.
    It will start the session selected in the sessions tree.
    """
    tree = STATE.get("sessions_tree")
    if not tree:
        return
    sel = tree.selection()
    if not sel:
        messagebox.showinfo("Select", "Select a session first.")
        return
    name = sel[0]
    sessions = load_sessions()
    if name not in sessions:
        messagebox.showerror("Error", "Session not found.")
        return
    info = sessions[name]
    start_session_thread_from_selection(name, info)

def download_current_session_csv():
    """
    Copy the CSV of the last-run session (STATE['session_name']) to a user-chosen location.
    """
    sname = STATE.get("session_name")
    if not sname:
        messagebox.showinfo("No session", "No session has run yet.")
        return
    fname = f"{ATTENDANCE_PREFIX}{sname}.csv"
    if not os.path.exists(fname):
        messagebox.showerror("Not found", f"{fname} not found.")
        return
    dst = filedialog.asksaveasfilename(defaultextension=".csv", initialfile=fname)
    if not dst:
        return
    try:
        shutil.copyfile(fname, dst)
        messagebox.showinfo("Saved", f"Saved {dst}")
        log(f"Saved CSV {dst}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save: {e}")
        log(f"❌ Failed to save CSV: {e}")

# Students helper functions 
def get_student_list():
    """Return list of student folder names under STUDENTS_DIR."""
    if not os.path.exists(STUDENTS_DIR):
        return []
    return sorted([d for d in os.listdir(STUDENTS_DIR) if os.path.isdir(os.path.join(STUDENTS_DIR, d))])

def refresh_student_combobox(combo):
    combo['values'] = get_student_list()
    combo.set('')

def remove_student_folder(name, combo=None):
    if not name:
        messagebox.showinfo("Select", "Choose student.")
        return
    if not messagebox.askyesno("Confirm", f"Delete student '{name}' folder?"):
        return
    folder = os.path.join(STUDENTS_DIR, name)
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            log(f"🗑️ Student '{name}' removed.")
            if combo:
                refresh_student_combobox(combo)
        except Exception as e:
            log(f"❌ Remove failed: {e}")
    else:
        log("⚠️ Student folder not found.")

def start_update_embeddings():
    """
    Run external encode generator if available in a background thread.
    """
    if not EXT_EMBEDDINGS:
        messagebox.showinfo("Missing", "EncodeGenerator.manage_embeddings not available.")
        return
    def worker():
        log("🔄 Running manage_embeddings() ...")
        try:
            ext_manage_embeddings()
            log("✅ Embeddings updated.")
        except Exception as e:
            log(f"❌ manage_embeddings error: {e}")
    threading.Thread(target=worker, daemon=True).start()

# GUI Entrypoint 
def main():
    ensure_dirs()
    root = build_gui()
    STATE["root"] = root
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root))
    root.mainloop()

def on_close(root):
    """
    Graceful shutdown handler for GUI close event.
    Stops attendance/student capture and releases camera if open.
    """
    if STATE.get("attendance_running"):
        if not messagebox.askyesno("Exit", "Attendance is running. Exit anyway?"):
            return
        STATE["attendance_stop"] = True
        time.sleep(0.5)
    # stop student capture if running
    if STATE.get("student_capture_running"):
        STATE["student_capture_running"] = False
        time.sleep(0.2)
    # release any camera
    try:
        if STATE.get("student_cap"):
            try:
                STATE["student_cap"].release()
            except Exception:
                pass
    except Exception:
        pass
    # destroy any OpenCV windows
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass

if __name__ == "__main__":
    main()
