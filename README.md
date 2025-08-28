# Golf Swing Analyzer

This project analyzes golf swings and classifies them as either **Pro** or **Amateur** using machine learning and pose estimation.

---

## Installation & Setup

1. Clone the repository:  
   ```bash
   git clone https://github.com/mcapriotti1/golf-swing-analyzer.git
   cd golf-swing-analyzer
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt:
3. Run the app locally
  ```bash
   python app.py

![Demo Screenshot](static/images/golf_demo.gif)
--

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
- Extracted landmark data **(x, y, z, visibility, presence)** for each of the listed body parts from **30** evenly spaced frames per video.

- Code for getting the landmark data from a video.

```python
landmarks = create_landmarks("example_video.mp4")
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
