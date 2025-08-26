import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from moviepy import VideoFileClip
import time
from pathlib import Path
import os
from moviepy import ImageSequenceClip
import json
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2




""" ------------------------------ EXTRACTING LANDMARK DATA --------------------------------------------- """

def trim_video(video_path, start_time, end_time, output_path=None):
    import time
    import os

    start = float(start_time)
    end = float(end_time)

    if end - start > 30:
      return None
    
    if output_path is None:
        timestamp = int(time.time() * 1000)
        output_dir = "website/trimmed_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
    
    with VideoFileClip(video_path) as video:
        video_duration = video.duration
        # Ensure end_time does not exceed video duration
        if end > video_duration:
            print(f"Warning: Requested end_time {end_time} exceeds video duration {video_duration}. Adjusting end_time.")
            end = video_duration
    
        if start < 0:
            print(f"Warning: Requested start_time {start} is less than 0. Adjusting start_time.")
            start = 0
        
        trimmed = video.subclipped(start, end)
        trimmed.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    return output_path

# def old_draw_landmarks(video_path, output_dir="main-project/website/static/landmarks_drawn_videos"):
#     os.makedirs(output_dir, exist_ok=True)
#     base_filename = os.path.basename(video_path)
#     output_path = os.path.join(output_dir, base_filename)

#     # Load Mediapipe models
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose()
#     mp_drawing = mp.solutions.drawing_utils

#     # Define a frame processing function
#     def annotate_frame(frame: np.ndarray) -> np.ndarray:
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         results = pose.process(image_rgb)

#         if results.pose_landmarks:
#             mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#         return cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

#     # Load the video
#     clip = VideoFileClip(video_path)

#     # Apply frame transformation
#     annotated_clip = clip.image_transform(annotate_frame)

#     # Save the new video
#     annotated_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

#     pose.close()
#     return f"landmarks_drawn_videos/{base_filename}"

def create_landmarks(video_path, num_frames = 30):
    # Setup MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode


    model_path = r"C:\Users\Micha\Golf_Swing\website\models\pose_landmarker_heavy.task"

    with open(model_path, "rb") as f:
      model_data = f.read()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    selected_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    selected_frames = set(selected_indices)

    landmarks = []


    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_idx in selected_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                timestamp_ms = int((frame_idx / fps) * 1000)
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

def draw_landmarks(video_path, output_dir="website/static/landmarks_drawn_videos", fast=False):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path = r"C:\Users\Micha\Golf_Swing\website\models\pose_landmarker_full.task"

    with open(model_path, "rb") as f:
        model_data = f.read()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, base_filename)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    annotated_frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    target_interval_ms = 300 if fast else 33 
    last_detection_time = -target_interval_ms
    last_result = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            if timestamp_ms - last_detection_time >= target_interval_ms:
                last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                last_detection_time = timestamp_ms

            if last_result and last_result.pose_landmarks:
                for part in KEY_BODY_PARTS:
                    idx = BODY_PARTS[part]
                    landmark = last_result.pose_landmarks[0][idx]
                    cx = int(landmark.x * width)
                    cy = int(landmark.y * height)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

            annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1

    cap.release()

    clip = ImageSequenceClip(annotated_frames, fps=fps)
    clip.write_videofile(output_path, codec="libx264", audio=False)

    return f"landmarks_drawn_videos/{base_filename}"



def cleanup_old_files(folder, max_age_minutes=10):
    now = time.time()
    max_age = max_age_minutes * 60  # seconds

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age:
                print(f"Deleting old file: {file_path}")
                os.remove(file_path)

def save_prediction(json_filename, file_path, prediction, confidence):
    # Create the dictionary entry

    filename = os.path.basename(file_path)
    new_entry = {
        filename: {
            "prediction": prediction,
            "confidence": confidence
        }
    }

    # Check if file exists
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Update or add the entry
    data.update(new_entry)

    # Save back to file
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)


TEN_MINUTES_MS = 10 * 60 * 1000 

def load_predictions(JSON_PATH):
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_predictions(data, JSON_PATH):
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

def clear_old_videos(JSON_PATH):
    data = load_predictions(JSON_PATH)
    now_ms = int(time.time() * 1000)

    def is_recent(filename):
        # Extract timestamp from filename, e.g., video_1756193983905.mp4
        try:
            timestamp_str = filename.split('_')[1].split('.')[0]
            timestamp = int(timestamp_str)
            age_ms = now_ms - timestamp
            return age_ms < TEN_MINUTES_MS
        except (IndexError, ValueError):
            # If format unexpected, keep the video just in case
            return True

    filtered_data = {fname: info for fname, info in data.items() if is_recent(fname)}

    if len(filtered_data) != len(data):
        save_predictions(filtered_data, JSON_PATH)
        print(f"Cleared old videos at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"No old videos to clear at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    


# def create_landmarks(video_path, output_path, num_frames=30, annotate = False):
#     landmarks = []

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error: Could not open input video.")
#         return
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     selected_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#     if not out.isOpened():
#         print("Error: Could not open output video writer.")
#         return

#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose()
#     mp_drawing = mp.solutions.drawing_utils

#     frame_idx = 0
#     selected_set = set(selected_indices)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_idx in selected_set:
#             image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = pose.process(image_rgb)

#             if results.pose_landmarks:
#                 if annotate:
#                     mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#                 frame_landmarks = [{
#                     'x': lm.x,
#                     'y': lm.y,
#                     'z': lm.z,
#                     'visibility': lm.visibility
#                 } for lm in results.pose_landmarks.landmark]

#                 landmarks.append(frame_landmarks)
#         if annotate:
#             out.write(frame)
#         frame_idx += 1

#     cap.release()
#     out.release()
#     return landmarks

""" CLEANING LANDMARK DATA (And Adding Velocitys/Joint Angles) """

BODY_PARTS = {
    "Nose": 0,
    "Left Eye Inner": 1,
    "Left Eye": 2,
    "Left Eye Outer": 3,
    "Right Eye Inner": 4,
    "Right Eye": 5,
    "Right Eye Outer": 6,
    "Left Ear": 7,
    "Right Ear": 8,
    "Mouth Left": 9,
    "Mouth Right": 10,
    "Left Shoulder": 11,
    "Right Shoulder": 12,
    "Left Elbow": 13,
    "Right Elbow": 14,
    "Left Wrist": 15,
    "Right Wrist": 16,
    "Left Pinky": 17,
    "Right Pinky": 18,
    "Left Index": 19,
    "Right Index": 20,
    "Left Thumb": 21,
    "Right Thumb": 22,
    "Left Hip": 23,
    "Right Hip": 24,
    "Left Knee": 25,
    "Right Knee": 26,
    "Left Ankle": 27,
    "Right Ankle": 28,
    "Left Heel": 29,
    "Right Heel": 30,
    "Left Foot Index": 31,
    "Right Foot Index": 32
}

KEY_BODY_PARTS = [
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Hip", "Right Hip",
    "Left Index", "Right Index",
    "Left Foot Index", "Right Foot Index",
    "Nose",
    "Left Knee", "Right Knee",
]

def compute_body_orientation_euler(frame):
    def get_point(name):
        idx = BODY_PARTS[name]
        return np.array([frame[idx]['x'], frame[idx]['y'], frame[idx]['z']])
    
    left_shoulder = get_point("Left Shoulder")
    right_shoulder = get_point("Right Shoulder")
    left_hip = get_point("Left Hip")
    right_hip = get_point("Right Hip")
    nose = get_point("Nose")
    
    # Create basis vectors for body local coordinate frame
    x_axis = right_shoulder - left_shoulder
    x_axis /= np.linalg.norm(x_axis)
    
    hip_vec = right_hip - left_hip
    hip_vec /= np.linalg.norm(hip_vec)
    
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2
    
    y_axis = mid_shoulder - mid_hip
    y_axis /= np.linalg.norm(y_axis)
    
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)
    
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    
    rot_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    
    r = R.from_matrix(rot_matrix)
    yaw, pitch, roll = r.as_euler('zyx', degrees=True)
    
    return [yaw, pitch, roll]

def angle_between(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_theta = dot / (norm1 * norm2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.acos(cos_theta)

def vector(p1, p2):
    return [p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']]

def compute_joint_angles(frame):
    angles = []
    
    def get_point(name):
        idx = BODY_PARTS[name]
        return np.array([frame[idx]['x'], frame[idx]['y'], frame[idx]['z']])
    
    def joint_angle(parent, joint, child):
        p = get_point(parent)
        j = get_point(joint)
        c = get_point(child)
        v1 = p - j
        v2 = c - j
        return angle_between(v1, v2)
    
    joints_to_compute = [
        ("Right Shoulder", "Right Elbow", "Right Index"),
        ("Left Hip", "Left Knee", "Left Foot Index"),
        ("Right Hip", "Right Knee", "Right Foot Index"),
        ("Left Shoulder", "Left Hip", "Left Foot Index"),
        ("Right Shoulder", "Right Hip", "Right Foot Index"),
        ("Left Elbow", "Left Shoulder", "Left Hip"),
        ("Right Elbow", "Right Shoulder", "Right Hip"),
    ]
    
    for parent, joint, child in joints_to_compute:
        angle = joint_angle(parent, joint, child)
        angles.append(angle)
    
    return angles

def compute_velocity(prev_frame, curr_frame):
    velocity = []
    for p1, p2 in zip(prev_frame, curr_frame):
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        dz = p2['z'] - p1['z']
        velocity.extend([dx, dy, dz])
    return velocity

def flatten_video(video):
    flat = []
    prevFrame = None
    for landmark in video:
        for frame in landmark:
            for part in KEY_BODY_PARTS:
                idx = BODY_PARTS[part]
                point = landmark[idx]
                flat.extend([point['x'], point['y'], point['z'], point['visibility'], point['presence']])
        flat.extend(compute_joint_angles(landmark))

        if prevFrame is not None:
            flat.extend(compute_velocity(prevFrame, frame))
        else:
            flat.extend([0] * 99)
    return flat


def normalize_landmarks(landmarks):
    normalized = []

    for frame in landmarks:
        if not frame:
            continue

        left_hip = frame[23]
        right_hip = frame[24]

        center_x = (left_hip['x'] + right_hip['x']) / 2
        center_y = (left_hip['y'] + right_hip['y']) / 2
        center_z = (left_hip['z'] + right_hip['z']) / 2

        frame_normalized = []
        for lm in frame:
            frame_normalized.append({
                'x': lm['x'] - center_x,
                'y': lm['y'] - center_y,
                'z': lm['z'] - center_z,
                'visibility': lm['visibility'],
                'presence': lm['presence']
            })
        normalized.append(frame_normalized)

    return normalized