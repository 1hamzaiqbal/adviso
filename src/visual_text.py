import re
from typing import List, Dict, Any, Optional, Tuple

def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z']+", text or "")

def _load_easyocr_reader() -> Optional[object]:
    try:
        import easyocr  # type: ignore
        # Force CPU (gpu=False) for stability across environments
        return easyocr.Reader(['en'], gpu=False)
    except Exception:
        return None

def _load_spellchecker():
    try:
        from spellchecker import SpellChecker  # type: ignore
        return SpellChecker()
    except Exception:
        return None

def _load_grammar_tool():
    try:
        import language_tool_python  # type: ignore
        return language_tool_python.LanguageTool('en-US')
    except Exception:
        return None

def extract_visual_text(video_path: str, fps: float = 1.0, max_frames: int = 90) -> Dict[str, Any]:
    """
    Extract on-screen text via OCR at a low sampling rate, then run spellcheck and simple grammar checks.
    Returns a dict with segments and summaries.
    """
    result: Dict[str, Any] = {
        'available': True,
        'segments': [],
        'misspellings_summary': {},
        'grammar_issues_summary': []
    }
    try:
        # Lazy imports to avoid heavy deps if not present
        reader = _load_easyocr_reader()
        if reader is None:
            return { 'available': False, 'reason': 'easyocr not installed', 'segments': [] }

        from .frame_extractor import read_frames  # local import
        frames, times, _ = read_frames(video_path, fps=fps)
        if max_frames and len(frames) > max_frames:
            frames = frames[:max_frames]
            times = times[:max_frames]

        spell = _load_spellchecker()
        tool = _load_grammar_tool()

        misspell_counts: Dict[str, int] = {}
        grammar_summary: List[Dict[str, Any]] = []

        for img, t in zip(frames, times):
            # OCR expects RGB; our frames are BGR from cv2
            try:
                import cv2  # type: ignore
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = img[:, :, ::-1]

            lines: List[Tuple] = []
            try:
                # each item: [bbox, text, confidence]
                lines = reader.readtext(rgb)
            except Exception:
                lines = []

            combined_texts: List[str] = []
            for item in lines:
                if len(item) >= 2:
                    txt = str(item[1]).strip()
                    if txt:
                        combined_texts.append(txt)

            if not combined_texts:
                continue

            text_block = " ".join(combined_texts)
            words = _tokenize_words(text_block)
            misspel: List[str] = []
            if spell is not None and words:
                try:
                    # Use lowercase for spellcheck
                    unknown = spell.unknown([w.lower() for w in words])
                    # Keep original words that match unknown lowercase variants
                    misspel = sorted(set([w for w in words if w.lower() in unknown]))
                    for w in misspel:
                        misspell_counts[w.lower()] = misspell_counts.get(w.lower(), 0) + 1
                except Exception:
                    misspel = []

            grammar_issues: List[Dict[str, Any]] = []
            if tool is not None and text_block:
                try:
                    matches = tool.check(text_block)
                    for m in matches:
                        suggestion = (m.replacements[0] if getattr(m, 'replacements', None) else '')
                        issue = {
                            'message': m.message,
                            'suggestion': suggestion
                        }
                        grammar_issues.append(issue)
                        # Keep a compact summary list
                        grammar_summary.append({
                            'time': float(t),
                            'message': m.message,
                            'suggestion': suggestion
                        })
                except Exception:
                    grammar_issues = []

            result['segments'].append({
                'time': float(t),
                'text': text_block,
                'words': words,
                'misspellings': misspel,
                'grammar_issues': grammar_issues
            })

        result['misspellings_summary'] = misspell_counts
        result['grammar_issues_summary'] = grammar_summary
        return result
    except Exception as e:
        return { 'available': False, 'reason': str(e), 'segments': [] }

