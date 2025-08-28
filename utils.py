import mediapipe as mp
import numpy as np
import math
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0" 
os.environ["OPENCV_VIDEOIO_PRIORITY_AVFOUNDATION"] = "0"
from moviepy import VideoFileClip
import time
from pathlib import Path
from moviepy import ImageSequenceClip
import json
import cv2
import imageio
from PIL import Image, ImageDraw
import subprocess

""" ------------------------------ EXTRACTING LANDMARK DATA --------------------------------------------- """

# def trim_video(video_path, start_time, end_time, output_path=None):
#     import time
#     import os

#     start = float(start_time)
#     end = float(end_time)

#     if end - start > 30:
#       return None
    
#     if output_path is None:
#         timestamp = int(time.time() * 1000)
#         output_dir = "static/trimmed_videos"
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
    
#     with VideoFileClip(video_path) as video:
#         video_duration = video.duration
#         # Ensure end_time does not exceed video duration
#         if end > video_duration:
#             print(f"Warning: Requested end_time {end_time} exceeds video duration {video_duration}. Adjusting end_time.")
#             end = video_duration
    
#         if start < 0:
#             print(f"Warning: Requested start_time {start} is less than 0. Adjusting start_time.")
#             start = 0
        
#         trimmed = video.subclipped(start, end)
#         trimmed.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
#     return output_path

# def trim_video(video_path, start_time, end_time, output_path=None):
#     start = float(start_time)
#     end = float(end_time)

#     if end - start > 30:
#         return None
    
#     if output_path is None:
#         timestamp = int(time.time() * 1000)
#         output_dir = "static/trimmed_videos"
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")
    
#     # Use ffmpeg to trim without re-encoding (super fast, low memory)
#     cmd = [
#         "ffmpeg", "-y",
#         "-ss", str(start),
#         "-i", video_path,
#         "-t", str(end-start),
#         "-c", "copy",
#         output_path
#     ]
#     subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     return output_path

def trim_video(video_path, start_time, end_time, output_path=None):
    start = float(start_time)
    end = float(end_time)

    if end - start > 30:
        return None  # enforce 30s max

    if output_path is None:
        timestamp = int(time.time() * 1000)
        output_dir = "static/trimmed_videos"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"video_{timestamp}.mp4")

    # Always re-encode for safety (guaranteed browser support)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(end - start),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",   # re-encode video
        "-c:a", "aac", "-b:a", "128k",                       # re-encode audio
        "-movflags", "+faststart",                           # make mp4 web-optimized
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return output_path

def copy_and_reencode_video(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(video_path))
    
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-y",  # overwrite if exists
        output_path
    ])
    
    return output_path


# def copy_and_reencode_video(video_path, output_dir):
#     """
#     Copies and re-encodes a video to ensure browser compatibility (H.264 MP4).
    
#     Args:
#         video_path (str): Path to the original video.
#         output_dir (str): Directory to save the re-encoded video.
    
#     Returns:
#         str: Full path to the re-encoded video.
#     """
#     print(video_path)
#     os.makedirs(output_dir, exist_ok=True)
#     base_filename = os.path.basename(video_path)
#     output_path = os.path.join(output_dir, base_filename)
    
#     with VideoFileClip(video_path) as clip:
#         clip.write_videofile(
#             output_path,
#             codec="libx264",
#             audio_codec="aac",
#             temp_audiofile="temp-audio.m4a",
#             remove_temp=True
#         )
    
#     return output_path

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
    model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
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


def extract_landmarks(video_path, fast=False):
    """
    Extracts pose landmarks from a video in a memory-efficient way.

    Args:
        video_path (str): Input video path.
        fast (bool): If True, processes fewer frames for speed.

    Returns:
        list[dict]: Each dict contains frame_index, timestamp (s), and landmarks [{x, y}].
                    Returns None if video is invalid.
    """
    import os, cv2, numpy as np, mediapipe as mp

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

    landmarks_list = []
    frame_idx = 0
    last_result = None
    target_interval_ms = 100 if fast else 20  # detect every 0.3s in fast mode

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp = frame_idx / fps
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(timestamp * 1000)

            # Only run detection on certain frames if fast
            if not fast or (timestamp_ms % target_interval_ms == 0):
                try:
                    last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
                except Exception as e:
                    print(f"Detection error at frame {frame_idx}: {e}")
                    last_result = None

            frame_landmarks = []
            if last_result and last_result.pose_landmarks:
                for idx in range(len(last_result.pose_landmarks[0])):
                    if BODY_PARTS_IDX[idx] in KEY_BODY_PARTS:
                        lm = last_result.pose_landmarks[0][idx]
                        frame_landmarks.append({"x": lm.x, "y": lm.y})

            landmarks_list.append({
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "landmarks": frame_landmarks
            })

            del frame, frame_rgb, mp_image
            frame_idx += 1

    cap.release()
    return landmarks_list



# def extract_landmarks(video_path):
#     """
#     Extracts pose landmarks from a video in a memory-efficient way.

#     Args:
#         video_path (str): Input video path.

#     Returns:
#         list[dict]: Each dict contains frame_index, timestamp (s), and landmarks [{x, y}].
#                     Returns None if video is invalid.
#     """
#     import os, cv2, numpy as np, mediapipe as mp

#     if not os.path.exists(video_path):
#         print(f"Video not found: {video_path}")
#         return None

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open video: {video_path}")
#         return None

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     if fps <= 0 or width <= 0 or height <= 0:
#         print(f"Invalid video metadata: {video_path}")
#         cap.release()
#         return None

#     # --- MediaPipe setup ---
#     mp_tasks = mp.tasks
#     BaseOptions = mp_tasks.BaseOptions
#     PoseLandmarker = mp_tasks.vision.PoseLandmarker
#     PoseLandmarkerOptions = mp_tasks.vision.PoseLandmarkerOptions
#     VisionRunningMode = mp_tasks.vision.RunningMode

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
#     with open(model_path, "rb") as f:
#         model_data = f.read()

#     options = PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_buffer=model_data),
#         running_mode=VisionRunningMode.VIDEO,
#         num_poses=1
#     )

#     landmarks_list = []
#     frame_idx = 0

#     with PoseLandmarker.create_from_options(options) as landmarker:
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break

#             timestamp = frame_idx / fps
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

#             try:
#                 result = landmarker.detect_for_video(mp_image, int(timestamp * 1000))
#             except Exception as e:
#                 print(f"Detection error at frame {frame_idx}: {e}")
#                 result = None

#             frame_landmarks = []
#             if result and result.pose_landmarks:
#                 for idx in range(len(result.pose_landmarks[0])):
#                     if BODY_PARTS_IDX[idx] in KEY_BODY_PARTS:
#                         lm = result.pose_landmarks[0][idx]
#                         frame_landmarks.append({"x": lm.x, "y": lm.y})  # normalized 0-1

#             landmarks_list.append({
#                 "frame_index": frame_idx,
#                 "timestamp": timestamp,
#                 "landmarks": frame_landmarks
#             })

#             del frame, frame_rgb, mp_image
#             frame_idx += 1

#     cap.release()
#     print(landmarks_list)
#     return landmarks_list


# import os
# import subprocess
# import tempfile
# import mediapipe as mp
# import cv2
# import numpy as np

# def draw_landmarks(video_path, output_dir="static/landmarks_drawn_videos_corrupt", fast=False):
#     """
#     Draws pose landmarks on a video using streaming (low memory) approach.

#     Args:
#         video_path (str): Input video path.
#         output_dir (str): Output directory.
#         fast (bool): If True, skips frames for faster processing.

#     Returns:
#         str: Path to the output video, or None if video invalid.
#     """
#     import os, subprocess, numpy as np, cv2, mediapipe as mp

#     # --- Check video validity ---
#     if not os.path.exists(video_path):
#         print(f"Video not found: {video_path}")
#         return None
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Cannot open video: {video_path}")
#         return None

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()
#     if fps <= 0 or width <= 0 or height <= 0:
#         print(f"Invalid video metadata: {video_path}")
#         return None

#     # --- MediaPipe setup ---
#     mp_tasks = mp.tasks
#     BaseOptions = mp_tasks.BaseOptions
#     PoseLandmarker = mp_tasks.vision.PoseLandmarker
#     PoseLandmarkerOptions = mp_tasks.vision.PoseLandmarkerOptions
#     VisionRunningMode = mp_tasks.vision.RunningMode

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
#     with open(model_path, "rb") as f:
#         model_data = f.read()

#     options = PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_buffer=model_data),
#         running_mode=VisionRunningMode.VIDEO,
#         num_poses=1
#     )

#     os.makedirs(output_dir, exist_ok=True)
#     base_filename = os.path.basename(video_path)
#     output_path = os.path.join(output_dir, base_filename)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     skip_interval = 3 if fast else 1
#     target_interval_ms = 300 if fast else 33

#     # --- Stream frames via FFmpeg ---
#     ffmpeg_cmd = [
#         "ffmpeg",
#         "-i", video_path,
#         "-f", "image2pipe",
#         "-pix_fmt", "bgr24",
#         "-vcodec", "rawvideo",
#         "-"
#     ]
#     proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, bufsize=10**7)

#     frame_idx = 0
#     last_detection_time = -target_interval_ms
#     last_result = None
#     frame_size = width * height * 3

#     with PoseLandmarker.create_from_options(options) as landmarker:
#         while True:
#             raw_frame = proc.stdout.read(frame_size)
#             if len(raw_frame) < frame_size:
#                 break  # end of video

#             frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)).copy()

#             # Skip frames for fast mode
#             if frame_idx % skip_interval != 0:
#                 out.write(frame)
#                 frame_idx += 1
#                 continue

#             # Pose detection
#             timestamp_ms = int((frame_idx / fps) * 1000)
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

#             if timestamp_ms - last_detection_time >= target_interval_ms:
#                 try:
#                     last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
#                 except Exception as e:
#                     print(f"Detection error at frame {frame_idx}: {e}")
#                     last_result = None
#                 last_detection_time = timestamp_ms

#             # Draw landmarks
#             if last_result and last_result.pose_landmarks:
#                 for part in KEY_BODY_PARTS:
#                     idx = BODY_PARTS[part]
#                     landmark = last_result.pose_landmarks[0][idx]
#                     cx = int(landmark.x * width)
#                     cy = int(landmark.y * height)
#                     cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

#             out.write(frame)

#             del frame, frame_rgb, mp_image
#             frame_idx += 1

#     out.release()
#     proc.stdout.close()
#     proc.wait()

#     if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
#         print("Failed to create output video.")
#         return None

#     print(f"Output video saved to: {output_path}")
#     return output_path



# def draw_landmarks(video_path, output_dir="static/landmarks_drawn_videos_corrupt", fast=False):
#     BaseOptions = mp.tasks.BaseOptions
#     PoseLandmarker = mp.tasks.vision.PoseLandmarker
#     PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
#     VisionRunningMode = mp.tasks.vision.RunningMode

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

#     with open(model_path, "rb") as f:
#         model_data = f.read()

#     options = PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_buffer=model_data),
#         running_mode=VisionRunningMode.VIDEO,
#         num_poses=1
#     )

#     os.makedirs(output_dir, exist_ok=True)
#     base_filename = os.path.basename(video_path)
#     output_path = os.path.join(output_dir, base_filename)

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # libx264 can be used too
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print("Total frames:", total_frames)

#     target_interval_ms = 300 if fast else 33 
#     last_detection_time = -target_interval_ms
#     last_result = None

#     with PoseLandmarker.create_from_options(options) as landmarker:
#         frame_idx = 0
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break

#             timestamp_ms = int((frame_idx / fps) * 1000)
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

#             if timestamp_ms - last_detection_time >= target_interval_ms:
#                 last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
#                 last_detection_time = timestamp_ms

#             # Draw landmarks
#             if last_result and last_result.pose_landmarks:
#                 for part in KEY_BODY_PARTS:
#                     idx = BODY_PARTS[part]
#                     landmark = last_result.pose_landmarks[0][idx]
#                     cx = int(landmark.x * width)
#                     cy = int(landmark.y * height)
#                     cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

#             # Write frame directly to output video
#             out.write(frame)
#             frame_idx += 1

#     cap.release()
#     out.release()

#     return f"static/landmarks_drawn_videos_corrupt/{base_filename}"

import subprocess

def copy_and_reencode_video(video_path, output_dir):
    print(video_path)
    """
    Re-encodes a video to H.264 MP4 for browser compatibility (memory efficient).
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory where the re-encoded video should be saved.
    
    Returns:
        str: Full path to the re-encoded video.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, base_filename)

    command = [
        "ffmpeg",
        "-y",  # overwrite if file exists
        "-i", video_path,
        "-c:v", "libx264",  # H.264 codec
        "-preset", "fast",  # balance speed and size
        "-c:a", "aac",      # audio codec
        "-movflags", "+faststart",  # better streaming in browsers
        output_path
    ]

    subprocess.run(command, check=True)

    return output_path




# def draw_landmarks(video_path, output_dir="static/landmarks_drawn_videos", fast=False):
#     BaseOptions = mp.tasks.BaseOptions
#     PoseLandmarker = mp.tasks.vision.PoseLandmarker
#     PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
#     VisionRunningMode = mp.tasks.vision.RunningMode

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")

#     # model_path = r"C:\Users\Micha\Golf_Swing\website\models\pose_landmarker_lite.task"

#     with open(model_path, "rb") as f:
#         model_data = f.read()

#     options = PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_buffer=model_data),
#         running_mode=VisionRunningMode.VIDEO,
#         num_poses=1
#     )

#     os.makedirs(output_dir, exist_ok=True)
#     base_filename = os.path.basename(video_path)
#     output_path = os.path.join(output_dir, base_filename)

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#     annotated_frames = []

#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     print("Total frames:", total_frames)

#     target_interval_ms = 300 if fast else 33 
#     last_detection_time = -target_interval_ms
#     last_result = None

#     with PoseLandmarker.create_from_options(options) as landmarker:
#         frame_idx = 0
#         while cap.isOpened():
#             success, frame = cap.read()
#             if not success:
#                 break

#             timestamp_ms = int((frame_idx / fps) * 1000)

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

#             if timestamp_ms - last_detection_time >= target_interval_ms:
#                 last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
#                 last_detection_time = timestamp_ms

#             if last_result and last_result.pose_landmarks:
#                 for part in KEY_BODY_PARTS:
#                     idx = BODY_PARTS[part]
#                     landmark = last_result.pose_landmarks[0][idx]
#                     cx = int(landmark.x * width)
#                     cy = int(landmark.y * height)
#                     cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

#             annotated_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             frame_idx += 1

#     cap.release()

#     clip = ImageSequenceClip(annotated_frames, fps=fps)
#     clip.write_videofile(output_path, codec="libx264", audio=False)

#     return f"landmarks_drawn_videos/{base_filename}"


# def create_landmarks(video_path, num_frames=30):
#     """
#     Extract pose landmarks from a video using MediaPipe Pose Landmarker.

#     Parameters:
#         video_path (str): Path to the video file.
#         num_frames (int): Number of evenly spaced frames to extract (default 30).

#     Returns:
#         list: List of frames, each containing a list of landmarks with x, y, z,
#               visibility, and presence values.
#     """
#     # MediaPipe setup
#     BaseOptions = mp.tasks.BaseOptions
#     PoseLandmarker = mp.tasks.vision.PoseLandmarker
#     PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
#     VisionRunningMode = mp.tasks.vision.RunningMode

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_heavy.task")
#     with open(model_path, "rb") as f:
#         model_data = f.read()

#     options = PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_buffer=model_data),
#         running_mode=VisionRunningMode.VIDEO,
#         num_poses=1
#     )

#     # Read video frames using imageio
#     reader = imageio.get_reader(video_path)
#     total_frames = reader.count_frames()
#     selected_indices = set(np.linspace(0, total_frames - 1, num=num_frames, dtype=int))

#     landmarks = []

#     with PoseLandmarker.create_from_options(options) as landmarker:
#         for i, frame in enumerate(reader):
#             if i in selected_indices:
#                 frame_rgb = np.array(frame)  # imageio returns RGB already
#                 mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
#                 timestamp_ms = int((i / reader.get_meta_data()['fps']) * 1000)
#                 result = landmarker.detect_for_video(mp_image, timestamp_ms)
#                 if result.pose_landmarks:
#                     frame_landmarks = [{
#                         'x': lm.x,
#                         'y': lm.y,
#                         'z': lm.z,
#                         'visibility': lm.visibility,
#                         'presence': lm.presence
#                     } for lm in result.pose_landmarks[0]]
#                     landmarks.append(frame_landmarks)

#     reader.close()
#     return landmarks


# def draw_landmarks(video_path, output_dir="static/landmarks_drawn_videos", fast=False):
#     """
#     Draw pose landmarks on a video and save the output.

#     Parameters:
#         video_path (str): Input video path.
#         output_dir (str): Directory to save annotated video.
#         fast (bool): If True, process frames less frequently for speed.

#     Returns:
#         str: Path to the saved annotated video.
#     """
#     BaseOptions = mp.tasks.BaseOptions
#     PoseLandmarker = mp.tasks.vision.PoseLandmarker
#     PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
#     VisionRunningMode = mp.tasks.vision.RunningMode

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     model_path = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
#     with open(model_path, "rb") as f:
#         model_data = f.read()

#     options = PoseLandmarkerOptions(
#         base_options=BaseOptions(model_asset_buffer=model_data),
#         running_mode=VisionRunningMode.VIDEO,
#         num_poses=1
#     )

#     os.makedirs(output_dir, exist_ok=True)
#     base_filename = os.path.basename(video_path)
#     output_path = os.path.join(output_dir, base_filename)

#     reader = imageio.get_reader(video_path)
#     fps = reader.get_meta_data()['fps']
#     width, height = reader.get_meta_data()['size']

#     annotated_frames = []

#     target_interval_ms = 300 if fast else 33
#     last_detection_time = -target_interval_ms
#     last_result = None
#     square_size = 9

#     with PoseLandmarker.create_from_options(options) as landmarker:
#         for i, frame in enumerate(reader):
#             timestamp_ms = int((i / fps) * 1000)

#             frame_rgb = np.array(frame)
#             mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

#             if timestamp_ms - last_detection_time >= target_interval_ms:
#                 last_result = landmarker.detect_for_video(mp_image, timestamp_ms)
#                 last_detection_time = timestamp_ms

#             if last_result and last_result.pose_landmarks:
#                 for part in KEY_BODY_PARTS:
#                     idx = BODY_PARTS[part]
#                     landmark = last_result.pose_landmarks[0][idx]
#                     cx = int(landmark.x * width)
#                     cy = int(landmark.y * height)
#                     frame_rgb[cy-square_size:cy+square_size, cx-square_size:cx+square_size] = [0, 255, 0]  # simple green square

#             annotated_frames.append(frame_rgb)

#     reader.close()
#     clip = ImageSequenceClip(annotated_frames, fps=fps)
#     clip.write_videofile(output_path, codec="libx264", audio=False)

#     return f"landmarks_drawn_videos/{base_filename}"


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

def cleanup_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def save_prediction(json_filename, file_path, prediction, confidence):

    filename = os.path.basename(file_path)
    new_entry = {
        filename: {
            "prediction": prediction,
            "confidence": confidence
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
        TEN_MINUTES_MS = 10 * 60 * 1000 
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