import numpy as np
import cv2

def _hist_feat(frame_bgr):
    hist = cv2.calcHist([frame_bgr], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def _clip_embed(frame_bgr, device='cpu'):
    try:
        import torch, clip
        from PIL import Image
        model, preprocess = clip.load('ViT-B/32', device=device, download=True)
        img = Image.fromarray(frame_bgr[:,:,::-1])
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(inp)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).detach().cpu().numpy()
    except Exception:
        return None

def compute_deltas(frames_bgr, method='hist', device='cpu'):
    deltas = []
    prev_feat = None
    for f in frames_bgr:
        if method == 'clip':
            feat = _clip_embed(f, device=device)
            if feat is None:
                feat = _hist_feat(f)
        else:
            feat = _hist_feat(f)
        if prev_feat is not None:
            if feat.ndim == 1 and prev_feat.ndim == 1 and feat.shape == prev_feat.shape:
                d = np.linalg.norm(feat - prev_feat)
            else:
                denom = (np.linalg.norm(feat)*np.linalg.norm(prev_feat) + 1e-6)
                d = 1.0 - float(np.dot(feat, prev_feat) / denom)
            deltas.append(float(d))
        prev_feat = feat
    return np.array(deltas, dtype='float32')

def cut_rate_series(deltas, fps, window_s=2.0, thresh='zscore'):
    if len(deltas) == 0:
        return np.array([])
    x = np.array(deltas, dtype='float32')
    if thresh == 'zscore':
        mu, sd = float(x.mean()), float(x.std() + 1e-6)
        events = (x > (mu + sd)).astype('float32')
    else:
        t = float(np.percentile(x, 75))
        events = (x > t).astype('float32')
    window = max(1, int(fps * window_s))
    kernel = np.ones(window, dtype='float32') / float(window)
    rate = np.convolve(events, kernel, mode='same') * fps
    return rate

def pacing_score_series(cut_rate, f_star=1.5, lam=2.0):
    return np.exp(-lam * (cut_rate - f_star)**2).astype('float32')
