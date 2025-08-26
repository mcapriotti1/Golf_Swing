import os
import json
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from collections import Counter
from utils import flatten_video, trim_video, draw_landmarks, normalize_landmarks, create_landmarks, cleanup_old_files, save_prediction, clear_old_videos, cleanup_folder
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'website/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
import joblib

# Load your model somewhere here (before first request or globally)
model = joblib.load("website/models/golf_swing_model.pkl")
JSON_PATH = "website/static/predictions.json"

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith('.mp4')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        cleanup_old_files("website/trimmed_videos", max_age_minutes=1)
        cleanup_old_files("website/static/landmarks_drawn_videos", max_age_minutes=2)

        # check if the post request has the file part
        if 'video' not in request.files:
            return "No file part", 400

        file = request.files['video']
        start = request.form["start"]
        end = request.form["end"]
        speed = request.form["model_type"]
        print("*" * 100)
        print(speed)

        if file.filename == '':
            return "No selected file", 400

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            trimmed = trim_video(filepath, start, end)
            if not trimmed:
                return "Your video is too long, upload a video under 30 seconds."

            # Run your pipeline here
            if speed == "lite":
                fast = True
            else:
                fast = False
            drawn_video_path = draw_landmarks(trimmed, fast=fast)
            landmarks = create_landmarks(trimmed)
            landmarks = normalize_landmarks(landmarks)

            if not landmarks:
                return "Your body could not be detected, try uploading a clearer video.", 400

            flattened_landmarks = np.array(flatten_video(landmarks))
            if len(flattened_landmarks) != 67530:
                return "Video too short, or your body could not be detected, try uploading a clearer video.", 400

            array = flattened_landmarks.reshape(1, -1)

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

            save_prediction(JSON_PATH, trimmed, final_prediction, confidence)
            clear_old_videos(JSON_PATH)
            cleanup_folder("website/uploads")
            
            trimmed_filename = os.path.basename(trimmed)
            return redirect(url_for('show_result', video_id=trimmed_filename))

        else:
            return "Unsupported file type. Please upload an mp4 video.", 400

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

    annotated_video_url = f"landmarks_drawn_videos/{video_id}"

    return render_template(
        'result.html',
        prediction=prediction,
        confidence=confidence,
        annotated_video_url=annotated_video_url
    )

if __name__ == '__main__':
    app.run(debug=True)
