import cv2
import numpy as np

def read_frames(path, fps=2, max_frames=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(native_fps // fps), 1)

    frames, times = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frames.append(frame)
            times.append(t)
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames, times, native_fps
