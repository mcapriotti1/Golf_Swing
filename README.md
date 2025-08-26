# Golf Swing Analyzer

This project analyzes golf swings and classifies them as either **Pro** or **Amateur** using machine learning and pose estimation.

---

## Dataset

- Collected around 100 golf swing videos:
  - 50 Pro swings
  - 50 Amateur swings
- Videos were labeled manually.

---

## Pose Extraction

- Used **MediaPipe Pose Landmarker (Heavy)** to extract landmarks.
- Extracted 30 evenly spaced frames per video.
- Key body parts used for analysis:

KEY_BODY_PARTS = [
"Left Shoulder", "Right Shoulder",
"Left Elbow", "Right Elbow",
"Left Hip", "Right Hip",
"Left Index", "Right Index",
"Left Foot Index", "Right Foot Index",
"Nose",
"Left Knee", "Right Knee",
]


- Positions normalized relative to **left hip**.
- Calculated **velocities** for each body part.
- Computed specific **joint angles**:


joints_to_compute = [
("Right Shoulder", "Right Elbow", "Right Index"),
("Left Hip", "Left Knee", "Left Foot Index"),
("Right Hip", "Right Knee", "Right Foot Index"),
("Left Shoulder", "Left Hip", "Left Foot Index"),
("Right Shoulder", "Right Hip", "Right Foot Index"),
("Left Elbow", "Left Shoulder", "Left Hip"),
("Right Elbow", "Right Shoulder", "Right Hip"),
]


- Included **presence** and **visibility** data from MediaPipe.

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
