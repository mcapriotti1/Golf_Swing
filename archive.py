import time
import os
from moviepy import VideoFileClip
import cv2
import mediapipe as mp
import numpy as np
import random
from utils import flatten_video, KEY_BODY_PARTS, BODY_PARTS
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

"""==================================================================================================== 
  All the functions in this file would work but require more than 500mb of memory to use for most videos ===================================================================================================="""

# Memory Heavy Trim Video
def trim_video(video_path, start_time, end_time, output_path=None):

    start = float(start_time)
    end = float(end_time)

    if end - start > 30:
      return None
    
    if output_path is None:
        timestamp = int(time.time() * 1000)
        output_dir = "static/trimmed_videos"
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

def draw_landmarks(video_path, output_dir="static/landmarks_drawn_videos", fast=False):
    """
    Draws pose landmarks on a video using a memory-efficient streaming approach.

    Args:
        video_path (str): Input video path.
        output_dir (str): Output directory.
        fast (bool): If True, skips frames for faster processing.

    Returns:
        str: Path to the output video, or None if video invalid.
    """
    import os, cv2, mediapipe as mp

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        print(f"Invalid video metadata: {video_path}")
        cap.release()
        return None

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

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, base_filename)

    # Resize frames to reduce memory usage
    resize_width = 640
    resize_height = int(height * (resize_width / width))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (resize_width, resize_height))

    skip_interval = 3 if fast else 1
    target_interval_frames = int(fps * 0.3) if fast else 1  # detect every 0.3s in fast mode

    frame_idx = 0
    last_result = None

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Resize to lower memory footprint
            frame = cv2.resize(frame, (resize_width, resize_height))

            # Skip frames for fast mode
            if frame_idx % skip_interval != 0:
                out.write(frame)
                frame_idx += 1
                continue

            # Pose detection
            if frame_idx % target_interval_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                try:
                    last_result = landmarker.detect_for_video(mp_image, int((frame_idx / fps) * 1000))
                except Exception as e:
                    print(f"Detection error at frame {frame_idx}: {e}")
                    last_result = None
                del frame_rgb, mp_image

            # Draw landmarks
            if last_result and last_result.pose_landmarks:
                for part in KEY_BODY_PARTS:
                    idx = BODY_PARTS[part]
                    lm = last_result.pose_landmarks[0][idx]
                    cx = int(lm.x * resize_width)
                    cy = int(lm.y * resize_height)
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

            out.write(frame)
            del frame
            frame_idx += 1

    cap.release()
    out.release()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print("Failed to create output video.")
        return None

    print(f"Output video saved to: {output_path}")
    return output_path


# Drawing Landmarks onto video and saving file (not work due to heavy memory usage)
def old_draw_landmarks(video_path, output_dir="static/landmarks_drawn_videos_corrupt", fast=False):
    """
    Draws pose landmarks on a video using streaming (low memory) approach.

    Args:
        video_path (str): Input video path.
        output_dir (str): Output directory.
        fast (bool): If True, skips frames for faster processing.

    Returns:
        str: Path to the output video, or None if video invalid.
    """
    import os, subprocess, numpy as np, cv2, mediapipe as mp

    # --- Check video validity ---
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or width <= 0 or height <= 0:
        print(f"Invalid video metadata: {video_path}")
        return None

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

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, base_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    skip_interval = 3 if fast else 1
    target_interval_ms = 300 if fast else 33

    # --- Stream frames via FFmpeg ---
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**7)

    frame_idx = 0
    last_detection_time = -target_interval_ms
    last_result = None
    frame_size = width * height * 3

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            raw_frame = proc.stdout.read(frame_size)
            if len(raw_frame) < frame_size:
                break  # end of video

            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)).copy()

            # Skip frames for fast mode
            if frame_idx % skip_interval != 0:
                out.write(frame)
                frame_idx += 1
                continue

            # Pose detection
            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            if timestamp_ms - last_detection_time >= target_interval_ms:
                try:
                    last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as e:
                    print(f"Detection error at frame {frame_idx}: {e}")
                    last_result = None
                last_detection_time = timestamp_ms

            # Draw landmarks
            if last_result and last_result.pose_landmarks:
                for part in KEY_BODY_PARTS:
                    idx = BODY_PARTS[part]
                    landmark = last_result.pose_landmarks[0][idx]
                    cx = int(landmark.x * width)
                    cy = int(landmark.y * height)
                    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

            out.write(frame)

            del frame, frame_rgb, mp_image
            frame_idx += 1

    out.release()
    proc.stdout.close()
    proc.wait()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print("Failed to create output video.")
        return None

    print(f"Output video saved to: {output_path}")
    return output_path

def set_seeds(seed=np.random.randint(0, 10000)):
    random.seed(seed)     
    np.random.seed(seed)    
    return seed

def train_random_forest(X, y):
    seed = set_seeds()
    print("\n=== Random Forest Classifier ===")
    X_flat = np.array([flatten_video(video) for video in X])
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.2, stratify=y, random_state=seed
    )

    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    return model, y_test, y_pred