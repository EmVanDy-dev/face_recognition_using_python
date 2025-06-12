import cv2
import face_recognition
import numpy as np
import pickle
import datetime
import requests
import tkinter as tk
from tkinter import messagebox
from threading import Thread
import os
import json

# Load known face data
with open("C:/Users/Mao Piseth/Downloads/face_data.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# Your deployed Apps Script Web App URL
GOOGLE_SCRIPT_URL = "https://script.google.com/macros/s/AKfycby3KPrDLLIa2jzAodMEso57H5tVJiUlZWmOnBnk8_5AnUFbNleOxEkLTbPIbSpoRv3s/exec"

# Recognition threshold
threshold = 0.45

# Selected action state
selected_action = ["Check In"]  # default

# Cache file to store daily submissions
CACHE_FILE = "submission_cache.json"

# Load or initialize submission cache (dict with structure: {date: {name: [modes]}})
def load_submission_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        today_data = data.get(today, {})
        # Ensure today's data is a dict; if not, reset it
        if not isinstance(today_data, dict):
            today_data = {}
        return today_data  # returns dict: {name: [modes]}
    return {}

# Save cache
def save_submission_cache(daily_data):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    data = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
    data[today] = daily_data
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f)

# Show a pop-up message
def show_confirmation(message):
    def _popup():
        messagebox.showinfo("Attendance Info", message)
    root.after(0, _popup)

# Face recognition function
def recognize_and_send():
    video_capture = cv2.VideoCapture(0)

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    zone_w, zone_h = 300, 300
    zone_x = (frame_width - zone_w) // 2
    zone_y = (frame_height - zone_h) // 2

    daily_submissions = load_submission_cache()  # {name: [modes]}
    alerted_names = set()  # To prevent repeated alerts in one session

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            best_distance = 1.0

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2

            if zone_x <= face_center_x <= zone_x + zone_w and zone_y <= face_center_y <= zone_y + zone_h:
                if known_encodings:
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(distances)
                    best_distance = distances[best_match_index]

                    if best_distance < threshold:
                        name = known_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                if name != "Unknown":
                    user_modes = daily_submissions.get(name, [])

                    # If user already has this mode today
                    if selected_action[0] in user_modes:
                        if (name, selected_action[0]) not in alerted_names:
                            show_confirmation(f"{name}, you already {selected_action[0].lower()}ed today!")
                            alerted_names.add((name, selected_action[0]))
                    else:
                        timestamp = datetime.datetime.now()
                        date = timestamp.strftime("%Y-%m-%d")
                        time = timestamp.strftime("%H:%M:%S")
                        data = {
                            "name": name,
                            "date": date,
                            "time": time,
                            "mode": selected_action[0]
                        }

                        print(f"Data being sent: {data}")

                        try:
                            response = requests.post(GOOGLE_SCRIPT_URL, data=data)
                            response_text = response.text.strip()
                            if response_text == "Success":
                                # Update cache
                                if name in daily_submissions:
                                    daily_submissions[name].append(selected_action[0])
                                else:
                                    daily_submissions[name] = [selected_action[0]]
                                save_submission_cache(daily_submissions)

                                print(f"[INFO] Sent to Google Sheet: {data}")
                                show_confirmation(f"{name}, your {selected_action[0]} is saved!")
                                alerted_names.add((name, selected_action[0]))
                            else:
                                print(f"[WARN] Unexpected response: {response_text}")
                        except Exception as e:
                            print("Failed to send data:", e)

        cv2.rectangle(frame, (zone_x, zone_y), (zone_x + zone_w, zone_y + zone_h), (255, 255, 0), 2)
        cv2.putText(frame, "Place your face here", (zone_x, zone_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        cv2.imshow('Face Recognition Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# GUI setup
def start_gui():
    def set_action(action):
        selected_action[0] = action
        print(f"Selected action: {action}")

    root.title("Attendance System")
    root.geometry("400x150")
    root.resizable(False, False)

    tk.Label(root, text="Select Option", font=("Arial", 14)).pack(pady=5)

    btn_frame = tk.Frame(root)
    btn_frame.pack()

    tk.Button(btn_frame, text="Check In", width=15, command=lambda: set_action("Check In")).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_frame, text="Check Out", width=15, command=lambda: set_action("Check Out")).pack(side=tk.LEFT, padx=10)

    tk.Button(root, text="Start Camera", width=25, command=lambda: Thread(target=recognize_and_send).start()).pack(pady=10)

# Launch the GUI
root = tk.Tk()
start_gui()
root.mainloop()
