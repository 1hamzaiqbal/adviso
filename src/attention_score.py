import numpy as np

def saliency_concentration(sal_map, center_bias=True):
    h, w = sal_map.shape
    total = sal_map.sum() + 1e-6
    if center_bias:
        yy, xx = np.mgrid[0:h,0:w]
        cx, cy = w/2, h/2
        sigma = min(w,h)/6
        gauss = np.exp(-(((xx-cx)**2 + (yy-cy)**2)/(2*sigma**2)))
        score = float((sal_map * gauss).sum() / total)
    else:
        score = float(np.percentile(sal_map, 95))
    return max(0.0, min(1.0, score))

def combine_scores(sal_series, motion_series, clip_series=None, clip_delta=0.0,
                   w_sal=0.5, w_motion=0.25, w_clip=0.15, pacing_series=None, w_pace=0.1,
                   age_group=None):
    """
    Combine different attention scores into a unified attention curve.
    
    Args:
        sal_series: List of saliency scores
        motion_series: List of motion scores
        clip_series: Optional list of CLIP scores
        clip_delta: CLIP delta value
        w_sal: Weight for saliency (overridden by age_group if provided)
        w_motion: Weight for motion (overridden by age_group if provided)
        w_clip: Weight for CLIP (overridden by age_group if provided)
        pacing_series: Optional list of pacing scores
        w_pace: Weight for pacing (overridden by age_group if provided)
        age_group: Optional age group key to use age-specific weights and time decay
        
    Returns:
        tuple: (curve list, overall score)
    """
    # Import age group config if age_group is provided
    if age_group is not None:
        from .age_groups import get_age_group_config
        age_config = get_age_group_config(age_group)
        w_sal = age_config["weights"]["saliency"]
        w_motion = age_config["weights"]["motion"]
        w_clip = age_config["weights"]["clip"]
        w_pace = age_config["weights"]["pacing"]
        time_decay_start = age_config["time_decay"]["start"]
        time_decay_end = age_config["time_decay"]["end"]
    else:
        time_decay_start = 1.2
        time_decay_end = 1.0
    
    S = np.array(sal_series, dtype='float32')
    M = np.array(motion_series, dtype='float32')
    if clip_series is None:
        C = np.full_like(S, 0.5)
        w_clip = 0.0
    else:
        C = np.array(clip_series, dtype='float32')
    if pacing_series is None:
        P = np.ones_like(S) * 0.7
        w_pace = 0.0
    else:
        P = np.array(pacing_series, dtype='float32')

    if S.max() > 0: S = S / (S.max()+1e-6)
    if M.max() > 0: M = M / (M.max()+1e-6)
    if C.max() > 0: C = C / (C.max()+1e-6)
    if P.max() > 0: P = np.clip(P, 0, 1)

    n = len(S)
    weights_time = np.linspace(time_decay_start, time_decay_end, n)
    curve = (w_sal*S + w_motion*M + w_clip*C + w_pace*P) * weights_time
    curve = np.clip(curve, 0, 1)
    overall = float(curve[:min(5, n)].mean()*0.6 + curve.mean()*0.4)
    return curve.tolist(), overall
