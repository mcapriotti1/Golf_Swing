import matplotlib
matplotlib.use("Agg")
import os
import json
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from collections import Counter
from utils import flatten_video, trim_video, normalize_landmarks, create_landmarks, cleanup_old_files, save_prediction, clear_old_videos, cleanup_folder, copy_and_reencode_video, extract_landmarks
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
        speed = request.form["model_type"]

        if file.filename == '':
            cleanup_folder("uploads")
            return render_template("upload.html", error="File Error, make sure to upload a mp4 video.")

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            print("-" * 30, "Trimming Video", "-" * 30)
            trimmed = trim_video(filepath, start, end)
            cleanup_folder("uploads")
            if not trimmed:
                return render_template("upload.html", error="Your video is too long, upload a video under 30 seconds.")

            # Run your pipeline here
            if speed == "lite":
                fast = True
            else:
                fast = False
            print("-" * 30, "Creating Landmarks", "-" * 30)
            landmarks = create_landmarks(trimmed)
            landmarks = normalize_landmarks(landmarks)

            if not landmarks:
                return render_template("upload.html", error="Your body could not be detected, try uploading a clear video of the swing")
            
            flattened_landmarks = np.array(flatten_video(landmarks))
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

            print("-" * 30, "Drawing Landmarks", "-" * 30)
            # drawn_video_path = draw_landmarks(trimmed, fast=fast)
            print("COPYING AND RENCODING")
            # copy_and_reencode_video(drawn_video_path, "static/landmarks_drawn_videos")
            landmarks_data = extract_landmarks(trimmed, fast)
            import json
            with open("static/video_landmarks.json", "w") as f:
                json.dump(landmarks_data, f)
            save_prediction(JSON_PATH, trimmed, final_prediction, confidence)
            clear_old_videos(JSON_PATH)
            
            trimmed_filename = os.path.basename(trimmed)
            return redirect(url_for('show_result', video_id=trimmed_filename))

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

    annotated_video_url = f"trimmed_videos/{video_id}"

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        annotated_video_url=annotated_video_url
    )

if __name__ == '__main__':
    app.run(debug=True)
