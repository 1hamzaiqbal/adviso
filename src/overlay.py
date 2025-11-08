import cv2
import numpy as np
from .saliency import heatmap_on_frame

def write_overlay_video(frames_bgr, sal_maps, out_path, fps=24):
    if not frames_bgr or not sal_maps:
        raise ValueError("Empty frames or saliency maps")
    
    # Ensure minimum fps for video playback (at least 1 fps)
    fps = max(1.0, float(fps))
    
    try:
        from moviepy.editor import ImageSequenceClip
        # Use moviepy for more reliable video writing
        h, w = frames_bgr[0].shape[:2]
        overlay_frames = []
        
        for f, sm in zip(frames_bgr, sal_maps):
            # Ensure saliency map matches frame dimensions
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            
            out = heatmap_on_frame(f, sm, alpha=0.6)
            
            # Ensure output is uint8 BGR, then convert to RGB for moviepy
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            
            # Convert BGR to RGB for moviepy
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            overlay_frames.append(out_rgb)
        
        # Create clip and write video
        clip = ImageSequenceClip(overlay_frames, fps=fps)
        try:
            clip.write_videofile(out_path, codec='libx264', audio=False, verbose=False, logger=None, preset='medium')
        except Exception as e:
            # Try with different codec if libx264 fails
            try:
                clip.write_videofile(out_path, codec='mpeg4', audio=False, verbose=False, logger=None)
            except Exception:
                raise RuntimeError(f"Failed to write video with moviepy: {e}")
        finally:
            clip.close()
        
    except ImportError:
        # Fallback to OpenCV if moviepy not available
        h, w = frames_bgr[0].shape[:2]
        
        # Try different codecs for better compatibility
        codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
        vw = None
        for codec in codecs:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
            if vw.isOpened():
                break
            if vw:
                vw.release()
            vw = None
        
        if vw is None or not vw.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {out_path}")
        
        for f, sm in zip(frames_bgr, sal_maps):
            # Ensure saliency map matches frame dimensions
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            
            out = heatmap_on_frame(f, sm, alpha=0.6)
            
            # Ensure output is uint8 BGR
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            
            vw.write(out)
        
        vw.release()
