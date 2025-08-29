import mediapipe as mp
import numpy as np
import math
import os
import time
import json
import cv2
import shutil
import subprocess
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0" 
os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "0"

""" ====================================================================================================
                                            VIDEO UTILITIES
   ==================================================================================================== """

# def ensure_mp4(video_path:str) -> str:
#     """
#     Re-encode a video to MP4 (H.264 video + AAC audio).

#     Args:
#         video_path (str): Input video file.

#     Returns:
#         str: Path to converted MP4 file.
#     """

#     timestamp = int(time.time() * 1000)
#     output_dir = "static/converted_videos"
#     os.makedirs(output_dir, exist_ok=True)
#     converted_path = os.path.join(output_dir, f"converted_{timestamp}.mp4")

#     cmd = [
#         "ffmpeg", "-y",
#         "-i", video_path,
#         "-c:v", "libx264",
#         "-c:a", "aac",
#         "-strict", "experimental",
#         converted_path
#     ]
#     subprocess.run(cmd, check=True)

#     return converted_path

# def mov_trim_video(video_path: str, start_time: float, end_time: float, output_path: str = None) -> str:
#     """
#     Trim a MOV/MP4 video between start and end timestamps.

#     Args:
#         video_path (str): Path to source video.
#         start_time (float): Start time in seconds.
#         end_time (float): End time in seconds.
#         output_path (str, optional): Output path for trimmed video.

#     Returns:
#         str | None: Trimmed video path, or None if duration exceeds 30s.
#     """
#     start = float(start_time)
#     end = float(end_time)
#     duration = end - start

#     if duration > 30:
#         return None
    
#     if output_path is None:
#         timestamp = int(time.time() * 1000)
#         output_dir = "static/trimmed_videos"
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")

#     cmd = [
#         "ffmpeg", "-y",
#         "-ss", str(start),
#         "-i", video_path,
#         "-t", str(duration),
#         "-map_metadata", "0",
#         "-c", "copy",
#         output_path
#     ]
    
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     return output_path

def copy_video(video_path: str, start, end, output_path: str = None) -> str:
    """
    Copy a MOV/MP4 video to a safe location (no trimming).

    Args:
        video_path (str): Path to source video.
        output_path (str, optional): Destination path for copied video.

    Returns:
        str: Path to copied video.
    """

    start = float(start)
    end = float(end)

    if end - start > 30:
        return None
    
    if output_path is None:
        timestamp = int(time.time() * 1000)
        output_dir = "static/uploaded_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")

    shutil.move(video_path, output_path)  # preserves metadata & is efficient
    return output_path


""" ====================================================================================================
                                        LANDMARK EXTRACTION
==================================================================================================== """


def create_landmarks(video_path, num_frames=30, start_time: float = 0, end_time: float = None):
    """
    Extract pose landmarks from a video using MediaPipe Pose Landmarker.
    Fast version: skips unneeded frames without decoding them.
    """
    import cv2, numpy as np, os, mediapipe as mp

    # --- MediaPipe setup ---
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
    with open(model_path, "rb") as f:
        model_data = f.read()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    # --- Open video ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames - 1
    end_frame = min(end_frame, total_frames - 1)

    # --- Precompute frames to process ---
    selected_indices = sorted(set(np.linspace(start_frame, end_frame, num=num_frames, dtype=int)))
    landmarks = []

    current_idx = 0
    next_idx_to_process = selected_indices.pop(0) if selected_indices else None

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and next_idx_to_process is not None:
            # Skip frames until the next one we need
            while current_idx < next_idx_to_process:
                if not cap.grab():  # just skip, don't decode
                    break
                current_idx += 1

            success, frame = cap.read()  # decode the frame we need
            if not success:
                break

            # Resize for memory efficiency
            h, w = frame.shape[:2]
            new_w = 640
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))

            # Convert and detect landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((current_idx / fps) * 1000)
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

            # Move to the next frame we want
            current_idx += 1
            next_idx_to_process = selected_indices.pop(0) if selected_indices else None

    cap.release()
    return landmarks


def extract_landmarks(video_path, start, end, fast=False):
    """
    Memory-efficient pose extraction for long videos.
    Only processes needed frames and resizes frames for speed.
    """
    import cv2, numpy as np, os, mediapipe as mp

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(float(start) * fps)
    end_frame = int(float(end) * fps)
    end_frame = min(end_frame, total_frames - 1)

    # --- MediaPipe setup ---
    mp_tasks = mp.tasks
    BaseOptions = mp_tasks.BaseOptions
    PoseLandmarker = mp_tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp_tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp_tasks.vision.RunningMode

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
    with open(model_path, "rb") as f:
        model_data = f.read()

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_data),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    # --- Determine frame indices to process ---
    interval_fast = 0.4
    interval_slow = 0.2

    if fast:
        num_frames = int((end_frame - start_frame) / (fps * interval_fast))
    else:
        num_frames = int((end_frame - start_frame) / (fps * interval_slow))

    selected_indices = sorted(set(np.linspace(start_frame, end_frame, num=num_frames, dtype=int)))
    landmarks_list = []

    current_idx = 0
    next_idx_to_process = selected_indices.pop(0) if selected_indices else None

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened() and next_idx_to_process is not None:
            # Skip frames without decoding
            while current_idx < next_idx_to_process:
                if not cap.grab():
                    break
                current_idx += 1

            success, frame = cap.read()  # decode only the frame we need
            if not success:
                break

            # Resize frame for memory/CPU efficiency
            h, w = frame.shape[:2]
            new_w = 640
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))

            # Convert and detect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((current_idx / fps) * 1000)

            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"Detection error at frame {current_idx}: {e}")
                result = None

            frame_landmarks = []
            if result and result.pose_landmarks:
                for idx in range(len(result.pose_landmarks[0])):
                    if BODY_PARTS_IDX[idx] in KEY_BODY_PARTS:
                        lm = result.pose_landmarks[0][idx]
                        frame_landmarks.append({"x": lm.x, "y": lm.y})

            landmarks_list.append({
                "frame_index": current_idx,
                "timestamp": current_idx / fps,
                "landmarks": frame_landmarks
            })

            # Move to next target frame
            current_idx += 1
            next_idx_to_process = selected_indices.pop(0) if selected_indices else None

            # Free memory
            del frame, frame_rgb, mp_image, result, frame_landmarks

    cap.release()
    return landmarks_list

""" ====================================================================================================
                    LANDMARK POST-PROCESSING (Angles, Velocity, Normalization)
==================================================================================================== """

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

BODY_PARTS_IDX = {v: k for k, v in BODY_PARTS.items()}

KEY_BODY_PARTS = [
    "Left Shoulder", "Right Shoulder",
    "Left Elbow", "Right Elbow",
    "Left Hip", "Right Hip",
    "Left Index", "Right Index",
    "Left Foot Index", "Right Foot Index",
    "Nose",
    "Left Knee", "Right Knee",
]

def angle_between(v1: list, v2: list) -> float:
    """Compute angle between two 3D vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_theta = dot / (norm1 * norm2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.acos(cos_theta)

def vector(p1: list, p2: list) -> list:
    """Return vector from point p1 → p2."""
    return [p2['x'] - p1['x'], p2['y'] - p1['y'], p2['z'] - p1['z']]

def compute_joint_angles(frame: list) -> list:
    """Compute key joint angles for one frame of landmarks."""
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

def compute_velocity(prev_frame: list, curr_frame: list) -> list:
    """Compute velocity (dx, dy, dz) of all landmarks between two frames."""
    velocity = []
    for p1, p2 in zip(prev_frame, curr_frame):
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        dz = p2['z'] - p1['z']
        velocity.extend([dx, dy, dz])
    return velocity

def flatten_video(landmarks: list) -> list:
    """Flatten video landmark data into a single numeric vector (x,y,z,visibility,presence + angles + velocity)."""
    flat = []
    prevFrame = None
    for landmark in landmarks:
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


def normalize_landmarks(landmarks: list) -> list:
    """Center landmarks on hip midpoint to normalize position across frames."""
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

""" ====================================================================================================
                                        FILE MANAGEMENT
==================================================================================================== """

def cleanup_old_files(folder: str, max_age_minutes: int = 10) -> None:
    """Delete files in folder older than max_age_minutes."""
    now = time.time()
    max_age = max_age_minutes * 60

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age:
                print(f"Deleting old file: {file_path}")
                os.remove(file_path)

def cleanup_folder(folder: str) -> None:
    """Remove all files inside folder (non-recursive)."""
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_prediction(json_filename: str, file_path: str, prediction: float, confidence: float, start: float, end: float, mov: bool):
    """Save prediction results for a video into a JSON file."""
    filename = os.path.basename(file_path)
    new_entry = {
        filename: {
            "prediction": prediction,
            "confidence": confidence,
            "start": float(start),
            "end": float(end),
            "mov": bool(mov),
        }
    }

    if os.path.exists(json_filename):
        with open(json_filename, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data.update(new_entry)

    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=2)

def load_predictions(JSON_PATH: str) -> dict:
    """Load predictions JSON if exists, else return {}."""
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_predictions(data: dict, JSON_PATH: str) -> None:
    """Write predictions dict to JSON file."""
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

def append_landmarks_to_json(filename: str, landmarks_data: dict, json_path="static/video_landmarks.json") -> None:
    """Append landmarks data for one video into a shared JSON file."""
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    existing_data[filename] = landmarks_data

    with open(json_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"Landmarks for {filename} appended to {json_path}")

def clear_old_videos(JSON_PATH: str) -> None:
    """Remove video entries older than 10 minutes from predictions JSON."""
    data = load_predictions(JSON_PATH)
    now_ms = int(time.time() * 1000)

    def is_recent(filename):
        TEN_MINUTES_MS = 60 * 1000 * 10
        try:
            timestamp_str = filename.split('_')[1].split('.')[0]
            timestamp = int(timestamp_str)
            age_ms = now_ms - timestamp
            return age_ms < TEN_MINUTES_MS
        except (IndexError, ValueError):
            return True

    filtered_data = {fname: info for fname, info in data.items() if is_recent(fname)}

    if len(filtered_data) != len(data):
        save_predictions(filtered_data, JSON_PATH)
        print(f"Cleared old videos at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"No old videos to clear at {time.strftime('%Y-%m-%d %H:%M:%S')}")