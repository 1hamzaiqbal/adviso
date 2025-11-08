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
                   w_sal=0.5, w_motion=0.3, w_clip=0.2):
    import numpy as np
    S = np.array(sal_series, dtype='float32')
    M = np.array(motion_series, dtype='float32')
    if clip_series is None:
        C = np.full_like(S, 0.5)
        w_clip = 0.0
    else:
        C = np.array(clip_series, dtype='float32')
    if S.max() > 0: S = S / (S.max()+1e-6)
    if M.max() > 0: M = M / (M.max()+1e-6)
    if C.max() > 0: C = C / (C.max()+1e-6)

    n = len(S)
    weights_time = np.linspace(1.2, 1.0, n)
    curve = (w_sal*S + w_motion*M + w_clip*C) * weights_time
    curve = np.clip(curve, 0, 1)

    overall = float(curve[:min(5, n)].mean()*0.6 + curve.mean()*0.4)
    return curve.tolist(), overall
