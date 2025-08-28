import time
import os
from moviepy import VideoFileClip
import cv2
import mediapipe as mp

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

# Old Create Landmarks (Not work since file frames corrupted during download)
def create_landmarks(video_path, num_frames=31, start_time=None):
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
    # if start_time is None:
    #     start_frame = 0
    # else:
    #     start_frame = int(start_time * fps)
    #     if start_frame >= total_frames:
    #         start_frame = total_frames - 1
    # print(start_frame)
    # print(total_frames)

    # total_frames -= start_frame
    landmarks = []

    # timestamps = np.linspace(0, duration, num=num_frames, endpoint=False)
    selected_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    selected_frames = set(selected_indices)


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


# Drawing Landmarks onto video and saving file (not work due to heavy memory usage)
def draw_landmarks(video_path, output_dir="static/landmarks_drawn_videos_corrupt", fast=False):
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