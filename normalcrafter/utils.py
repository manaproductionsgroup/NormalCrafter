from typing import Union, List
import tempfile
import numpy as np
import PIL.Image
import matplotlib.cm as cm
import mediapy
import torch
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from decord import VideoReader, cpu


def read_video_frames(video_path, process_length, target_fps, max_res):
    print("==> processing video: ", video_path)
    vid = VideoReader(video_path, ctx=cpu(0))
    print("==> original video shape: ", (len(vid), *vid.get_batch([0]).shape[1:]))
    original_height, original_width = vid.get_batch([0]).shape[1:3]
    
    if max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)
    else:
        height = original_height
        width = original_width

    vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

    fps = vid.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid), stride))
    print(
        f"==> downsampled shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}, with stride: {stride}"
    )
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    print(
        f"==> final processing shape: {len(frames_idx), *vid.get_batch([0]).shape[1:]}"
    )
    frames = vid.get_batch(frames_idx).asnumpy().astype(np.uint8)
    frames = [PIL.Image.fromarray(x) for x in frames]

    return frames, fps

def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    crf: int = 0, #lossless
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]

    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
    return output_video_path

def extract_unique_frames(video_path, output_folder, threshold=0.95):
    """
    Extracts unique frames from a video using SSIM similarity.

    threshold = 1.0 (very strict, only identical frames ignored)
    threshold = 0.95 (recommended)
    threshold = 0.80 (more sensitive)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video.")

    prev_gray = None
    frame_index = 0
    unique_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            # First frame is always unique
            cv2.imwrite(f"{output_folder}/{frame_index:04d}.png", frame)
            prev_gray = gray
            frame_index += 1
            unique_count += 1
            continue

        # Compute SSIM between this frame and previous unique frame
        score, _ = ssim(gray, prev_gray, full=True)

        if score < threshold:
            # Frame is different enough → save it
            cv2.imwrite(f"{output_folder}/{frame_index:04d}.png", frame)
            prev_gray = gray
            unique_count += 1

        frame_index += 1

    cap.release()

    print(f"Extracted {unique_count} unique frames.")

def vis_sequence_normal(normals: np.ndarray):
    normals = normals.clip(-1., 1.)
    normals = normals * 0.5 + 0.5
    return normals
