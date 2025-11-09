import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from .frame_extractor import read_frames
from .saliency import spectral_residual_saliency
from .motion import motion_series
from .clip_scorer import score_frames_with_prompts
from .attention_score import saliency_concentration, combine_scores
from .overlay import write_overlay_video
from .scene_change import compute_deltas, cut_rate_series, pacing_score_series
from .score_interpreter import interpret_score
from .age_groups import get_pacing_for_age_group, AGE_GROUPS
from .brand_checks import evaluate_brand_consistency, parse_hex_palette

GOAL_PRESETS = {
    "hook": {"f_star": 0.5, "lambda": 0.5},
    "explainer": {"f_star": 0.3, "lambda": 1.2},
    "calm_brand": {"f_star": 0.2, "lambda": 1.0}
}

def main():
    ap = argparse.ArgumentParser(description='Ad Attention Analyzer (+ pacing)')
    ap.add_argument('--video', required=True, help='path to mp4')
    ap.add_argument('--out', default='./out', help='output directory')
    ap.add_argument('--fps', type=float, default=2.0, help='sampling fps')
    ap.add_argument('--use-clip', action='store_true', help='enable CLIP scoring')
    ap.add_argument('--goal', choices=list(GOAL_PRESETS.keys()), default='hook', help='goal preset')
    ap.add_argument('--lambda', dest='lam', type=float, default=None, help='override lambda')
    ap.add_argument('--scene-method', choices=['hist','clip'], default='hist', help='delta method')
    ap.add_argument('--age-group', choices=list(AGE_GROUPS.keys()), default='general',
                    help='target age group for scoring (affects weights, thresholds, and pacing preferences)')
    ap.add_argument('--brand-logo', default=None, help='path to brand logo image (optional)')
    ap.add_argument('--brand-colors', default=None, help='comma-separated brand hex colors (e.g. #FF0000,#00FF00)')
    ap.add_argument('--brand-terms', default=None, help='comma-separated brand keywords to expect in on-screen text')
    ap.add_argument('--ocr-text', action='store_true', help='run OCR-based message clarity checks (requires easyocr)')
    ap.add_argument('--logo-threshold', type=float, default=0.6, help='confidence threshold for logo detection (0-1)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames, times, native_fps = read_frames(args.video, fps=args.fps)
    if not frames:
        raise SystemExit('No frames extracted.')
    frame_times = list(times)

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

    deltas = compute_deltas(frames, method=args.scene_method)
    cut_rate = cut_rate_series(deltas, fps=args.fps, window_s=2.0, thresh='zscore')
    cut_rate_series_full = np.concatenate([[cut_rate[0] if len(cut_rate)>0 else 0.0], cut_rate]).tolist()

    # Get pacing preferences based on age group and goal
    age_pacing = get_pacing_for_age_group(args.age_group, args.goal)
    preset = GOAL_PRESETS[args.goal]
    # Use age group pacing preferences by default
    lam = args.lam if args.lam is not None else age_pacing['lambda']
    f_star = age_pacing['f_star']
    pacing_series = pacing_score_series(np.array(cut_rate_series_full, dtype='float32'), f_star=f_star, lam=lam).tolist()

    curve, overall = combine_scores(sal_scores, mot, clip_series, clip_delta, pacing_series=pacing_series, w_pace=0.2, age_group=args.age_group)

    arr = np.array(curve)
    peaks_idx = np.argsort(arr)[-3:][::-1].tolist()
    times = times[:len(curve)]
    key_moments = [{'time': float(times[i]), 'score': float(arr[i])} for i in peaks_idx]

    # Plots
    plt.figure(figsize=(8,3))
    plt.plot(times, curve, label='Attention')
    plt.xlabel('Time (s)'); plt.ylabel('Score (0..1)')
    plt.title('Predicted Attention Curve')
    plt.grid(True); plt.tight_layout()
    curve_path = os.path.join(args.out, 'attention_curve.png')
    plt.savefig(curve_path, dpi=150); plt.close()

    plt.figure(figsize=(8,3))
    tr = times[:len(cut_rate_series_full)]
    plt.plot(tr, cut_rate_series_full, label='Cut rate (cuts/sec)')
    plt.axhline(f_star, ls='--', label=f'Goal f*={f_star}')
    plt.xlabel('Time (s)'); plt.ylabel('Cuts/sec')
    plt.title('Editing Rhythm')
    plt.legend(); plt.grid(True); plt.tight_layout()
    rhythm_path = os.path.join(args.out, 'editing_rhythm.png')
    plt.savefig(rhythm_path, dpi=150); plt.close()

    plt.figure(figsize=(8,3))
    plt.plot(tr, pacing_series[:len(tr)], label='Pacing score')
    plt.xlabel('Time (s)'); plt.ylabel('Score (0..1)')
    plt.title('Goal-Adjusted Pacing Score')
    plt.grid(True); plt.tight_layout()
    pacing_path = os.path.join(args.out, 'pacing_score.png')
    plt.savefig(pacing_path, dpi=150); plt.close()

    overlay_path = os.path.join(args.out, 'overlay.mp4')
    write_overlay_video(frames, sal_maps, overlay_path, fps=max(2, int(args.fps)), deltas=deltas)

    first5s = float(np.mean(curve[:min(5, len(curve))]))
    avg_cut = float(np.mean(cut_rate)) if len(cut_rate)>0 else 0.0
    
    # Generate score interpretation with age group
    interpretation = interpret_score(overall, first5s, avg_cut, args.goal, f_star, age_group=args.age_group)

    brand_palette = parse_hex_palette(args.brand_colors)
    brand_terms = [t.strip() for t in (args.brand_terms or '').split(',') if t.strip()]
    brand_eval = evaluate_brand_consistency(
        frames=frames,
        times=frame_times,
        video_path=args.video,
        brand_logo_path=args.brand_logo,
        brand_colors=brand_palette,
        brand_terms=brand_terms,
        run_ocr=args.ocr_text or bool(brand_terms),
        logo_threshold=float(args.logo_threshold),
    )

    report = {
        'Goal': args.goal,
        'AgeGroup': args.age_group,
        'Pacing': {'f_star': f_star, 'lambda': lam},
        'OverallAttentionScore': round(float(overall), 3),
        'First5sRetention': round(first5s, 3),
        'AvgCutRate': avg_cut,
        'KeyMoments': key_moments,
        'FramesAnalyzed': len(frames),
        'SamplingFPS': args.fps,
        'UsedCLIP': bool(args.use_clip),
        'BrandInputs': {
            'brandLogoPath': args.brand_logo,
            'brandColors': brand_palette,
            'brandTerms': brand_terms,
            'ocrText': bool(args.ocr_text or brand_terms),
            'logoThreshold': float(args.logo_threshold),
        },
        'BrandEvaluation': brand_eval,
        'ScoreInterpretation': interpretation,
        'Files': {
            'overlay_video': overlay_path,
            'attention_curve_png': curve_path,
            'editing_rhythm_png': rhythm_path,
            'pacing_score_png': pacing_path
        }
    }
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
