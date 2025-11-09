"""Utility helpers for lightweight brand & safety critique heuristics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency but usually available via requirements.txt
    import cv2  # type: ignore
except Exception:  # pragma: no cover - guard for environments without cv2
    cv2 = None  # type: ignore


@dataclass
class LogoDetectionResult:
    available: bool
    detected: bool
    score: Optional[float]
    max_score: Optional[float]
    frame_index: Optional[int]
    time_sec: Optional[float]
    threshold: float
    reason: Optional[str] = None
    location: Optional[Tuple[int, int]] = None


HEX_ALLOWED = set("0123456789abcdefABCDEF")


def _parse_hex_color(value: str) -> Optional[Tuple[int, int, int]]:
    if not value:
        return None
    value = value.strip().lstrip("#")
    if len(value) not in (6, 8) or any(ch not in HEX_ALLOWED for ch in value):
        return None
    # Ignore alpha channel if provided
    value = value[:6]
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (r, g, b)


def parse_hex_palette(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    items = [item.strip() for item in raw.split(",") if item.strip()]
    valid: List[str] = []
    for item in items:
        rgb = _parse_hex_color(item)
        if rgb is not None:
            valid.append(f"#{item.strip().lstrip('#')[:6].upper()}")
    return valid


def _bgr_to_lab(color: Tuple[int, int, int]) -> np.ndarray:
    if cv2 is None:
        return np.array(color[::-1], dtype=np.float32)
    sample = np.uint8([[list(color[::-1])]])  # convert RGB -> BGR
    lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1).astype(np.float32)


def _frame_mean_lab(frame: np.ndarray) -> np.ndarray:
    if cv2 is None:
        # Approximate by RGB mean if cv2 missing
        mean_rgb = frame.reshape(-1, frame.shape[-1]).mean(axis=0)
        return mean_rgb[::-1]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0)


def evaluate_color_alignment(frames: Sequence[np.ndarray], palette_hex: Sequence[str]) -> Dict[str, Any]:
    if not palette_hex:
        return {'available': False, 'reason': 'No palette provided'}
    try:
        palette_lab = np.stack([_bgr_to_lab(_parse_hex_color(color) or (0, 0, 0)) for color in palette_hex])
    except Exception as exc:
        return {'available': False, 'reason': f'Failed to parse palette: {exc}'}

    distances: List[float] = []
    for frame in frames:
        lab = _frame_mean_lab(frame)
        dists = np.linalg.norm(palette_lab - lab, axis=1)
        distances.append(float(dists.min()))

    avg_dist = float(np.mean(distances)) if distances else float('inf')
    # Convert distance (0..~180) to similarity 0..1
    similarity = float(np.clip(1.0 - (avg_dist / 110.0), 0.0, 1.0))
    return {
        'available': True,
        'score': similarity,
        'avgDistanceLab': avg_dist,
        'palette': list(palette_hex),
    }


def _resize_logo_if_needed(logo: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if cv2 is None:
        return logo
    logo_h, logo_w = logo.shape[:2]
    frame_h, frame_w = target_shape
    if logo_h <= frame_h and logo_w <= frame_w:
        return logo
    scale = min(frame_h / max(logo_h, 1), frame_w / max(logo_w, 1)) * 0.9
    if scale <= 0:
        return logo
    new_w = max(1, int(logo_w * scale))
    new_h = max(1, int(logo_h * scale))
    return cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)


def detect_logo(frames: Sequence[np.ndarray], times: Sequence[float], logo_path: Optional[str], threshold: float = 0.6) -> LogoDetectionResult:
    if not logo_path:
        return LogoDetectionResult(False, False, None, None, None, None, threshold, reason='No logo provided')
    if cv2 is None:
        return LogoDetectionResult(False, False, None, None, None, None, threshold, reason='cv2 unavailable')
    logo = cv2.imread(logo_path)
    if logo is None:
        return LogoDetectionResult(False, False, None, None, None, None, threshold, reason='Unable to read logo file')

    best_score = -1.0
    best_idx: Optional[int] = None
    best_loc: Optional[Tuple[int, int]] = None
    logo_resized = _resize_logo_if_needed(logo, frames[0].shape[:2])
    logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
    for idx, frame in enumerate(frames):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_gray.shape[0] < logo_gray.shape[0] or frame_gray.shape[1] < logo_gray.shape[1]:
            continue
        res = cv2.matchTemplate(frame_gray, logo_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_idx = idx
            best_loc = max_loc

    if best_idx is None or best_score < 0:
        return LogoDetectionResult(True, False, 0.0, 0.0, None, None, threshold, reason='No viable match')

    confidence = float(np.clip((best_score - 0.3) / 0.5, 0.0, 1.0))
    time_sec = float(times[best_idx]) if best_idx < len(times) else None
    detected = bool(best_score >= threshold)
    return LogoDetectionResult(
        True,
        detected,
        confidence,
        float(best_score),
        best_idx,
        time_sec,
        threshold,
        location=best_loc,
    )


def analyze_text_segments(video_path: str, brand_terms: Sequence[str], run_ocr: bool) -> Dict[str, Any]:
    if not run_ocr and not brand_terms:
        return {'available': False, 'reason': 'OCR disabled'}
    try:
        from .visual_text import extract_visual_text
    except Exception as exc:  # pragma: no cover - import guard
        return {'available': False, 'reason': f'visual_text unavailable: {exc}'}

    result = extract_visual_text(video_path, fps=1.0, max_frames=90)
    if not result.get('available'):
        return {'available': False, 'reason': result.get('reason', 'OCR unavailable')}

    segments = result.get('segments') or []
    lower_terms = [term.lower() for term in brand_terms if term]
    mentions: Dict[str, int] = {term: 0 for term in lower_terms}
    for seg in segments:
        text = str(seg.get('text', '')).lower()
        for term in lower_terms:
            if term and term in text:
                mentions[term] += 1

    total_segments = len(segments)
    unique_terms_hit = sum(1 for term, count in mentions.items() if count > 0)
    coverage = (unique_terms_hit / max(len(lower_terms), 1)) if lower_terms else (1.0 if segments else 0.0)
    score = 0.4 + 0.6 * coverage
    if (result.get('misspellings_summary') or {}):
        score *= 0.85
    if (result.get('grammar_issues_summary') or []):
        score *= 0.9
    score = float(np.clip(score, 0.0, 1.0))

    return {
        'available': True,
        'score': score,
        'segments': total_segments,
        'brandTermHits': mentions,
        'misspellings': sorted((result.get('misspellings_summary') or {}).keys())[:5],
        'grammarIssues': (result.get('grammar_issues_summary') or [])[:5],
    }


def analyze_safety(frames: Sequence[np.ndarray]) -> Dict[str, Any]:
    if not frames:
        return {'available': False, 'reason': 'No frames'}
    gray_means: List[float] = []
    for frame in frames:
        if cv2 is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_means.append(float(gray.mean()))
        else:
            gray_means.append(float(frame.mean()))

    diffs = np.abs(np.diff(gray_means)) if len(gray_means) > 1 else np.array([0.0])
    flicker_metric = float(np.percentile(diffs, 95)) if len(diffs) else 0.0

    issues: List[str] = []
    score = 1.0
    min_val = min(gray_means)
    max_val = max(gray_means)
    if min_val < 20:
        issues.append('Many frames are extremely dark')
        score -= 0.2
    if max_val > 235:
        issues.append('Some frames are extremely bright')
        score -= 0.2
    if flicker_metric > 45:
        issues.append('High luminance flicker detected (may be uncomfortable)')
        score -= 0.3
    if len(diffs) and np.mean(diffs) > 25:
        issues.append('Average frame-to-frame change is high; review for visual safety')
        score -= 0.1

    score = float(np.clip(score, 0.0, 1.0))
    return {
        'available': True,
        'score': score,
        'issues': issues,
        'flickerMetric': flicker_metric,
        'luminanceRange': {'min': float(min_val), 'max': float(max_val)},
    }


def evaluate_brand_consistency(
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    video_path: str,
    brand_logo_path: Optional[str] = None,
    brand_colors: Optional[Sequence[str]] = None,
    brand_terms: Optional[Sequence[str]] = None,
    run_ocr: bool = False,
    logo_threshold: float = 0.6,
) -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    scores: List[float] = []
    flags: List[str] = []

    if brand_colors:
        color_result = evaluate_color_alignment(frames, brand_colors)
        components['colorPalette'] = color_result
        score = color_result.get('score')
        if isinstance(score, (int, float)):
            scores.append(float(score))
    elif brand_colors == []:
        components['colorPalette'] = {'available': False, 'reason': 'No palette provided'}

    logo_result = detect_logo(frames, times, brand_logo_path, threshold=logo_threshold)
    if isinstance(logo_result, LogoDetectionResult):
        components['logoDetection'] = asdict(logo_result)
    else:
        components['logoDetection'] = logo_result
    if isinstance(logo_result, LogoDetectionResult) and isinstance(logo_result.score, (int, float)):
        scores.append(float(logo_result.score))
        if logo_result.available and not logo_result.detected:
            flags.append('Logo not confidently detected in sample frames')
    elif isinstance(logo_result, dict):
        score = logo_result.get('score')
        if isinstance(score, (int, float)):
            scores.append(float(score))

    text_terms = list(brand_terms or [])
    text_result = analyze_text_segments(video_path, text_terms, run_ocr=run_ocr or bool(text_terms))
    components['messageClarity'] = text_result
    if isinstance(text_result.get('score'), (int, float)):
        scores.append(float(text_result['score']))
        if text_terms and sum(text_result.get('brandTermHits', {}).values()) == 0:
            flags.append('No brand terms detected in on-screen text')

    safety_result = analyze_safety(frames)
    components['safetyHeuristics'] = safety_result
    if isinstance(safety_result.get('score'), (int, float)):
        scores.append(float(safety_result['score']))
        if safety_result.get('issues'):
            flags.extend([f"Safety: {issue}" for issue in safety_result['issues']])

    overall = float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else None

    return {
        'overallScore': overall,
        'components': components,
        'flags': flags,
        'inputs': {
            'hasLogo': bool(brand_logo_path),
            'paletteProvided': bool(brand_colors),
            'brandTerms': text_terms,
            'ocrEnabled': run_ocr or bool(text_terms),
        },
    }

