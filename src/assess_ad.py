import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from .frame_extractor import read_frames
from .saliency import spectral_residual_saliency
from .motion import motion_series
from .clip_scorer import score_frames_with_prompts
from .attention_score import saliency_concentration, combine_scores
from .overlay import write_overlay_video

def main():
    ap = argparse.ArgumentParser(description='Ad Attention Analyzer')
    ap.add_argument('--video', required=True, help='path to mp4')
    ap.add_argument('--out', default='./out', help='output directory')
    ap.add_argument('--fps', type=float, default=2.0, help='sampling fps')
    ap.add_argument('--use-clip', action='store_true', help='enable CLIP scoring')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    frames, times, native_fps = read_frames(args.video, fps=args.fps)
    if not frames:
        raise SystemExit('No frames extracted.')

    sal_maps, sal_scores = [], []
    for f in frames:
        sm = spectral_residual_saliency(f)
        sal_maps.append(sm)
        sal_scores.append(saliency_concentration(sm))

    mot = motion_series(frames)

    clip_series = None
    clip_delta = 0.0
    if args.use_clip:
        clip_series, clip_delta = score_frames_with_prompts(frames)

    curve, overall = combine_scores(sal_scores, mot, clip_series, clip_delta)

    arr = np.array(curve)
    peaks_idx = np.argsort(arr)[-3:][::-1].tolist()
    times = times[:len(curve)]
    key_moments = [{"time": float(times[i]), "score": float(arr[i])} for i in peaks_idx]

    plt.figure(figsize=(8,3))
    plt.plot(times, curve, label='Attention')
    plt.xlabel('Time (s)'); plt.ylabel('Score (0..1)')
    plt.title('Predicted Attention Curve')
    plt.grid(True); plt.tight_layout()
    curve_path = os.path.join(args.out, 'attention_curve.png')
    plt.savefig(curve_path, dpi=150)
    plt.close()

    overlay_path = os.path.join(args.out, 'overlay.mp4')
    write_overlay_video(frames, sal_maps, overlay_path, fps=max(2, int(args.fps)))

    report = {
        "OverallAttentionScore": round(float(overall), 3),
        "First5sRetention": round(float(np.mean(curve[:min(5, len(curve))])), 3),
        "KeyMoments": key_moments,
        "FramesAnalyzed": len(frames),
        "SamplingFPS": args.fps,
        "UsedCLIP": bool(args.use_clip),
        "Files": {
            "overlay_video": overlay_path,
            "attention_curve_png": curve_path
        }
    }
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
