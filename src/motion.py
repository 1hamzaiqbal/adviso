import cv2
import numpy as np

def motion_energy(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    return np.mean(mag)

def motion_series(frames_bgr):
    series = []
    prev = None
    for f in frames_bgr:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is None:
            series.append(0.0)
        else:
            series.append(motion_energy(prev, g))
        prev = g
    arr = np.array(series, dtype='float32')
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr.tolist()
