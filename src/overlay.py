import cv2
import numpy as np
from .saliency import heatmap_on_frame

def _draw_scene_change_indicator(frame, frame_idx, deltas, threshold=None):
    """Draw scene change indicators on frame"""
    # Convert numpy array to list if needed
    if deltas is not None:
        if hasattr(deltas, 'tolist'):
            deltas = deltas.tolist()
        elif not isinstance(deltas, (list, tuple)):
            deltas = list(deltas)
    
    if deltas is None or len(deltas) == 0:
        return frame
    
    h, w = frame.shape[:2]
    
    # Calculate threshold if not provided (zscore method)
    if threshold is None:
        if len(deltas) > 0:
            mu = float(np.mean(deltas))
            sd = float(np.std(deltas) + 1e-6)
            threshold = mu + sd
        else:
            threshold = 0.0
    
    # Get current delta (frame_idx corresponds to deltas index)
    # First frame has no delta, so we use 0 or the first delta
    if frame_idx == 0:
        current_delta = 0.0
    elif frame_idx <= len(deltas):
        current_delta = float(deltas[frame_idx - 1])
    else:
        current_delta = float(deltas[-1]) if len(deltas) > 0 else 0.0
    
    # Draw vertical line indicator if scene change detected
    is_scene_change = current_delta > threshold
    if is_scene_change:
        # Draw thicker vertical line on the left side (more visible)
        cv2.line(frame, (10, 0), (10, h), (0, 255, 255), 5)  # Yellow line
        cv2.line(frame, (8, 0), (8, h), (0, 255, 255), 5)
        cv2.line(frame, (12, 0), (12, h), (0, 255, 255), 5)
        cv2.line(frame, (6, 0), (6, h), (0, 200, 255), 3)  # Additional highlight
        cv2.line(frame, (14, 0), (14, h), (0, 200, 255), 3)
    
    # Draw histogram/delta graph at the bottom
    graph_height = 100
    graph_y = h - graph_height - 10
    graph_x_start = 20
    graph_width = w - 40
    
    # Draw graph background (semi-transparent black) - always visible
    overlay = frame.copy()
    cv2.rectangle(overlay, (graph_x_start, graph_y), 
                  (graph_x_start + graph_width, graph_y + graph_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Draw delta values as a line graph
    if len(deltas) > 0:
        max_delta = max(float(np.max(deltas)), threshold * 1.5) if threshold > 0 else float(np.max(deltas)) + 1e-6
        min_delta = 0.0
        delta_range = max_delta - min_delta + 1e-6
        
        # Draw threshold line (more visible)
        thresh_y = graph_y + graph_height - int((threshold - min_delta) / delta_range * graph_height)
        cv2.line(frame, (graph_x_start, thresh_y), 
                (graph_x_start + graph_width, thresh_y), 
                (128, 128, 128), 2, cv2.LINE_AA)  # Gray threshold line
        
        # Add threshold label
        cv2.putText(frame, f"Threshold: {threshold:.3f}", 
                   (graph_x_start + graph_width - 150, thresh_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Draw delta line graph
        # Show deltas up to current frame (deltas[0] to deltas[frame_idx-1] for frame_idx > 0)
        num_deltas_to_show = min(frame_idx, len(deltas))
        if num_deltas_to_show > 0:
            points = []
            # Start from x=0 for first delta
            for i in range(num_deltas_to_show):
                x = graph_x_start + int((i / max(len(deltas), 1)) * graph_width)
                delta_val = float(deltas[i])
                y = graph_y + graph_height - int((delta_val - min_delta) / delta_range * graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                for i in range(len(points) - 1):
                    color = (0, 255, 255) if float(deltas[i]) > threshold else (255, 255, 255)
                    cv2.line(frame, points[i], points[i+1], color, 3, cv2.LINE_AA)
            
            # Draw current point (if we have a delta for this frame)
            if frame_idx > 0 and frame_idx <= len(deltas):
                curr_x = graph_x_start + int(((frame_idx - 1) / max(len(deltas), 1)) * graph_width)
                curr_y = graph_y + graph_height - int((current_delta - min_delta) / delta_range * graph_height)
                point_color = (0, 255, 255) if is_scene_change else (255, 255, 255)
                cv2.circle(frame, (curr_x, curr_y), 6, point_color, -1)
                cv2.circle(frame, (curr_x, curr_y), 8, (0, 0, 0), 2)  # Black outline for visibility
    
    # Add text label (more visible)
    label_text = f"Delta: {current_delta:.3f}"
    if is_scene_change:
        label_text += " [SCENE CHANGE!]"
        text_color = (0, 255, 255)  # Yellow for scene change
    else:
        text_color = (255, 255, 255)  # White otherwise
    
    # Draw text with background for better visibility
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (graph_x_start + 5, graph_y + 5), 
                  (graph_x_start + text_width + 15, graph_y + text_height + 15), 
                  (0, 0, 0), -1)
    cv2.putText(frame, label_text, (graph_x_start + 10, graph_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
    
    return frame

def write_overlay_video(frames_bgr, sal_maps, out_path, fps=24, deltas=None):
    if not frames_bgr or not sal_maps:
        raise ValueError("Empty frames or saliency maps")
    
    # Debug: print deltas info
    if deltas is not None:
        import sys
        print(f"Scene change visualization: deltas shape={getattr(deltas, 'shape', 'N/A')}, len={len(deltas) if hasattr(deltas, '__len__') else 'N/A'}", file=sys.stderr)
    
    # Ensure minimum fps for video playback (at least 1 fps)
    fps = max(1.0, float(fps))
    
    try:
        from moviepy.editor import ImageSequenceClip
        # Use moviepy for more reliable video writing
        h, w = frames_bgr[0].shape[:2]
        overlay_frames = []
        
        for idx, (f, sm) in enumerate(zip(frames_bgr, sal_maps)):
            # Ensure saliency map matches frame dimensions
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            
            out = heatmap_on_frame(f, sm, alpha=0.6)
            
            # Add scene change visualization
            if deltas is not None:
                try:
                    out = _draw_scene_change_indicator(out, idx, deltas)
                except Exception as e:
                    # If visualization fails, continue without it
                    import sys
                    print(f"Warning: Scene change visualization failed on frame {idx}: {e}", file=sys.stderr)
            
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
        
        for idx, (f, sm) in enumerate(zip(frames_bgr, sal_maps)):
            # Ensure saliency map matches frame dimensions
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            
            out = heatmap_on_frame(f, sm, alpha=0.6)
            
            # Add scene change visualization
            if deltas is not None:
                try:
                    out = _draw_scene_change_indicator(out, idx, deltas)
                except Exception as e:
                    # If visualization fails, continue without it
                    import sys
                    print(f"Warning: Scene change visualization failed on frame {idx}: {e}", file=sys.stderr)
            
            # Ensure output is uint8 BGR
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            
            vw.write(out)
        
        vw.release()
