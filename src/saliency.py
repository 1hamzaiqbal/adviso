import cv2
import numpy as np

def spectral_residual_saliency(frame_bgr):
    sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = sal.computeSaliency(frame_bgr)
    if not success:
        sal2 = cv2.saliency.StaticSaliencyFineGrained_create()
        success2, saliency_map = sal2.computeSaliency(frame_bgr)
        if not success2:
            h, w = frame_bgr.shape[:2]
            return np.zeros((h,w), dtype=np.float32)
    saliency_map = saliency_map.astype('float32')
    if saliency_map.max() > 0:
        saliency_map = saliency_map / saliency_map.max()
    return saliency_map

def heatmap_on_frame(frame_bgr, sal_map, alpha=0.6):
    # Ensure saliency map is 2D and matches frame dimensions
    if len(sal_map.shape) == 3:
        sal_map = sal_map[:, :, 0]  # Take first channel if 3D
    
    h, w = frame_bgr.shape[:2]
    if sal_map.shape[:2] != (h, w):
        sal_map = cv2.resize(sal_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] if needed
    if sal_map.max() > 1.0:
        sal_map = sal_map / 255.0
    elif sal_map.max() > 0:
        sal_map = sal_map / sal_map.max()
    
    # Convert to uint8 and apply colormap
    sal_uint8 = (sal_map * 255).astype(np.uint8)
    sal_color = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
    
    # Blend with frame (original formula for visibility)
    out = cv2.addWeighted(frame_bgr, 1.0, sal_color, alpha, 0)
    return out
