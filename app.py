import matplotlib
matplotlib.use("Agg")
import os
import json
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from collections import Counter
from utils import flatten_video, trim_video, normalize_landmarks, create_landmarks, cleanup_old_files, save_prediction, clear_old_videos, cleanup_folder, copy_and_reencode_video, extract_landmarks, ensure_mp4, mov_create_landmarks, mov_trim_video, append_landmarks_to_json
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
import joblib
import threading, time
import psutil, os

def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"[MEMORY] {note} RSS={mem_info.rss / (1024*1024):.2f} MB, VMS={mem_info.vms / (1024*1024):.2f} MB")

def monitor_memory():
    while True:
        log_memory_usage("Live Monitor")
        time.sleep(5)

threading.Thread(target=monitor_memory, daemon=True).start()

model = joblib.load("models/golf_swing_model.pkl")
JSON_PATH = "static/predictions.json"

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('.mp4', ".mov"))

def ends_with_mov(filename):
    return '.' in filename and filename.lower().endswith(".mov")

os.makedirs("static", exist_ok=True)
os.makedirs("static/trimmed_videos", exist_ok=True)
os.makedirs("static/landmarks_drawn_videos", exist_ok=True)
os.makedirs("static/landmarks_drawn_videos_corrupt", exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        cleanup_old_files("static/trimmed_videos", max_age_minutes=1)
        cleanup_old_files("static/landmarks_drawn_videos", max_age_minutes=2)
        cleanup_old_files("static/landmarks_drawn_videos_corrupt", max_age_minutes=1)

        print("-" * 30, "Downloading Video", "-" * 30)

        if 'video' not in request.files:
            cleanup_folder("uploads")
            return render_template("upload.html", error="File Error, make sure to upload a mp4 video.")

        file = request.files['video']
        start = request.form["start"]
        end = request.form["end"]
        end = str(float(end) - 0.1)
        speed = request.form["model_type"]

        if file.filename == '':
            cleanup_folder("uploads")
            return render_template("upload.html", error="File Error, make sure to upload a mp4 video.")

        if file and allowed_file(file.filename):
            mov = ends_with_mov(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            """ ------------- CODE FOR TRIMMING IF HAD MORE MEMORY ------------------- """
            # print("-" * 30, "Trimming Video", "-" * 30)
            video_path = ensure_mp4(filepath)
            print(video_path)
            # if mov:
            #     trimmed = mov_trim_video(video_path, start, end)
            # else:
            # trimmed = mov_trim_video(video_path, start, end)
            
            # if not trimmed:
            #     return render_template("upload.html", error="Your video is too long, upload a video under 30 seconds.")

            if speed == "lite":
                fast = True
            else:
                fast = False
            print("-" * 30, "Creating Landmarks", "-" * 30)
            print("HELLO")
            landmarks = mov_create_landmarks(video_path, start_time=float(start), end_time=float(end))
                # landmarks = create_landmarks(video_path, start_time=float(start))
            landmarks = normalize_landmarks(landmarks)

            if not landmarks:
                return render_template("upload.html", error="Your body could not be detected, try uploading a clear video of the swing")
            
            flattened_landmarks = np.array(flatten_video(landmarks))
            print("LENGTH:", len(flattened_landmarks))
            if len(flattened_landmarks) != 67530:
                return render_template("upload.html", error="Video too short, or your body could not be detected, try uploading a clearer video.")

            array = flattened_landmarks.reshape(1, -1)

            print("-" * 30, "Making Prediction", "-" * 30)
            prediction = model.predict(array)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(array)
                confidence = np.max(probs)
                final_prediction = model.classes_[np.argmax(probs)]
            else:
                final_prediction = Counter(prediction).most_common(1)[0][0]
                confidence = None

            if final_prediction == 1:
                final_prediction = "Pro"
            else:
                final_prediction = "Amateur"

            print("-" * 30, "Extracting Landmarks For Video", "-" * 30)
            
            # Draw on backend if have the memory

            # drawn_video_path = draw_landmarks(trimmed, fast=fast)

            landmarks_data = extract_landmarks(video_path, fast)
            filename = os.path.basename(video_path)

            append_landmarks_to_json(filename, landmarks_data)

            save_prediction(JSON_PATH, video_path, final_prediction, confidence, start, end, mov)
            clear_old_videos(JSON_PATH)
            
            return redirect(url_for('show_result', video_id=filename))

        else:
            cleanup_folder("uploads")
            return render_template("upload.html", error="Unsupported file type. Please upload an mp4 video.")

    return render_template('upload.html')

@app.route('/result/<video_id>')
def show_result(video_id):
    # Load predictions json
    if not os.path.exists(JSON_PATH):
        return "No predictions found.", 404

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    prediction_data = data.get(video_id)
    if not prediction_data:
        return "Prediction not found.", 404

    prediction = prediction_data['prediction']
    confidence = prediction_data['confidence']
    start = prediction_data['start']
    end = prediction_data['end']
    mov = prediction_data['mov']

    annotated_video_url = f"converted_videos/{video_id}"

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        annotated_video_url=annotated_video_url,
        start=start,
        end=end,
        mov=mov,
    )

if __name__ == '__main__':
    app.run(debug=True)
