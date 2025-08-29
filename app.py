import matplotlib
matplotlib.use("Agg")
import os
import json
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from collections import Counter
from utils import flatten_video, normalize_landmarks, cleanup_old_files, save_prediction, clear_old_videos, cleanup_folder, extract_landmarks, create_landmarks, append_landmarks_to_json, copy_video
from archive import draw_landmarks
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
import joblib
import logging
logging.basicConfig(level=logging.INFO)

model = joblib.load("models/golf_swing_model.pkl")
JSON_PATH = "static/predictions.json"
JSON_LANDMARKS_PATH = "static/video_landmarks.json"

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith(('.mp4', ".mov"))

def ends_with_mov(filename):
    return '.' in filename and filename.lower().endswith(".mov")

os.makedirs("static", exist_ok=True)
os.makedirs("static/uploaded_videos", exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # cleanup_old_files("static/converted_videos", max_age_minutes=1)
        cleanup_old_files("static/uploaded_videos", max_age_minutes=2)

        logging.info("Downloading Video")
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
            # video_path = ensure_mp4(filepath)
            copy = copy_video(filepath, float(start), float(end))
            cleanup_folder("uploads")

            # if mov:
            #     trimmed = mov_trim_video(video_path, start, end)
            # else:
            # trimmed = mov_trim_video(video_path, start, end)
            
            if not copy:
                return render_template("upload.html", error="Your video is too long, upload a video under 30 seconds.")

            if speed == "lite":
                fast = True
            else:
                fast = False
            print("-" * 30, "Creating Landmarks", "-" * 30)
            logging.info("Creating Landmarks")
            landmarks = create_landmarks(copy, start_time=float(start), end_time=float(end))
                # landmarks = create_landmarks(video_path, start_time=float(start))
            landmarks = normalize_landmarks(landmarks)

            if not landmarks:
                return render_template("upload.html", error="Your body could not be detected, try uploading a clear video of the swing")
            
            flattened_landmarks = np.array(flatten_video(landmarks))
            if len(flattened_landmarks) != 67530:
                return render_template("upload.html", error="Video too short, or your body could not be detected, try uploading a clearer video.")

            array = flattened_landmarks.reshape(1, -1)

            logging.info("Making Prediction")
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

            logging.info("Extracting Landmarks For Video")
            print("-" * 30, "Extracting Landmarks For Video", "-" * 30)
            
            # Draw on backend if have the memory

            # drawn_video_path = draw_landmarks(trimmed, fast=fast)

            landmarks_data = extract_landmarks(copy, start, end, fast)
            print(draw_landmarks(file, "static"))
            filename = os.path.basename(copy)

            append_landmarks_to_json(filename, landmarks_data)

            save_prediction(JSON_PATH, copy, final_prediction, confidence, start, end, mov)
            clear_old_videos(JSON_PATH)
            clear_old_videos(JSON_LANDMARKS_PATH)
            
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

    annotated_video_url = f"uploaded_videos/{video_id}"

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