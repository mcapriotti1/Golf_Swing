# Golf Swing Analyzer

This project analyzes golf swings and classifies them as either **Pro** or **Amateur** using machine learning and pose estimation.

---

## Dataset

- Collected 100 golf swing videos:
  - 50 Pro swings
  - 50 Amateur swings
- Videos were labeled manually.

---

## Pose Extraction

- Used **MediaPipe Pose Landmarker (Heavy)** to extract landmarks.
- Key body parts used for analysis:
```python
KEY_BODY_PARTS = [
"Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow","Left Hip", "Right Hip", "Left Index", 
"Right Index", "Left Foot Index", "Right Foot Index", "Nose", "Left Knee", "Right Knee" ]
```
- Extracted landmark data for each of the listed body parts from **30** evenly spaced frames per video. Each video is represented as a flattened feature array of length **67,530**, including **normalized landmark positions**, **velocities**, **joint angles**, and **visibility/presence** data across all frames.

```python
def create_landmarks(video_path, num_frames=30):
    """
    Extract pose landmarks from a video using MediaPipe Pose Landmarker.

    Parameters:
        video_path (str): Path to the video file.
        num_frames (int): Number of evenly spaced frames to extract (default 30).

    Returns:
        list: List of frames, each containing a list of landmarks with x, y, z,
                visibility, and presence values.
    """

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Load the pre-trained pose landmarker model
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_heavy.task")
    with open(model_path, "rb") as f:
        model_data = f.read()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    # Open video and select frames evenly spaced across the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    selected_frames = set(selected_indices)

    landmarks = []

    # Process video frames with MediaPipe
    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_idx in selected_frames:
                # Convert frame to RGB and prepare for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)

                # Detect pose landmarks
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.pose_landmarks:
                    frame_landmarks = [{
                        'x': lm.x,
                        'y': lm.y,
                        'z': lm.z,
                        'visibility': lm.visibility,
                        'presence': lm.presence
                    } for lm in result.pose_landmarks[0]]
                    landmarks.append(frame_landmarks)

            frame_idx += 1

    cap.release()
    return landmarks
```

- Normalized x, y, z positions relative to the left hip, calculated velocities for each body part (first frame initialized to 0), and computed the specified joint angles.

```python
joints_to_compute = [
  ("Right Shoulder", "Right Elbow", "Right Index"), ("Left Hip", "Left Knee", "Left Foot Index"),
  ("Right Hip", "Right Knee", "Right Foot Index"), ("Left Shoulder", "Left Hip", "Left Foot Index"),
  ("Right Shoulder", "Right Hip", "Right Foot Index"), ("Left Elbow", "Left Shoulder", "Left Hip"),
  ("Right Elbow", "Right Shoulder", "Right Hip") ]
```

---

## Model Training

- Used **Random Forest Classifier** from **scikit-learn**.
- Example training code:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

model = RandomForestClassifier(n_estimators=100, random_state=76)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```
