from __future__ import annotations
import os
import re
import time
import uuid
import tempfile
from typing import Any, Dict, Optional, Tuple


MAX_INLINE_BYTES = 15 * 1024 * 1024  # ~15MB inline limit for Vertex parts


def _init_vertex(project: str, location: str):
    try:
        from vertexai import init as vertex_init  # type: ignore
        vertex_init(project=project, location=location)
        return True, None
    except Exception as e:
        return False, f"vertexai init failed: {e}"


def _ensure_ffmpeg_in_path() -> None:
    from shutil import which
    if which("ffmpeg") is not None:
        return
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        ffmpeg_path = get_ffmpeg_exe()
        bin_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _compress_for_inline(video_bytes: bytes, content_type: str, max_seconds: float = 60.0) -> bytes:
    """Compress video to fit under MAX_INLINE_BYTES by trimming and downscaling.
    Returns new bytes (mp4). If compression fails, returns original bytes.
    """
    try:
        _ensure_ffmpeg_in_path()
        from moviepy.editor import VideoFileClip  # type: ignore
        with tempfile.TemporaryDirectory() as td:
            in_path = os.path.join(td, "in.mp4")
            with open(in_path, "wb") as f:
                f.write(video_bytes)
            with VideoFileClip(in_path) as clip:
                end = min(max_seconds, clip.duration or max_seconds)
                sub = clip.subclip(0, end)
                # Downscale to max width 480, keep aspect
                try:
                    w = sub.w or 0
                    if w and w > 480:
                        sub = sub.resize(width=480)
                except Exception:
                    pass
                out_path = os.path.join(td, "out.mp4")
                # Conservative bitrate + mono audio and lower fps
                sub.write_videofile(
                    out_path,
                    codec="libx264",
                    audio_codec="aac",
                    bitrate="350k",
                    audio_bitrate="48k",
                    fps=max(10, int(getattr(sub, 'fps', 24) or 24)),
                    verbose=False,
                    logger=None,
                )
                data = open(out_path, "rb").read()
                if len(data) <= MAX_INLINE_BYTES:
                    return data
                # Try harsher downscale
                out2 = os.path.join(td, "out2.mp4")
                try:
                    sub2 = sub.resize(width=320)
                except Exception:
                    sub2 = sub
                sub2.write_videofile(
                    out2,
                    codec="libx264",
                    audio_codec="aac",
                    bitrate="240k",
                    audio_bitrate="32k",
                    fps=max(8, int(getattr(sub, 'fps', 24) or 24)),
                    verbose=False,
                    logger=None,
                )
                data2 = open(out2, "rb").read()
                return data2 if len(data2) < len(data) else data
    except Exception:
        return video_bytes


def analyze_brand_vertex(
    video_bytes: bytes,
    filename: str,
    content_type: str,
    project: str,
    location: str = "us-central1",
    gcs_bucket: Optional[str] = None,
    brand_name: Optional[str] = None,
    brand_context: Optional[str] = None,
    max_seconds: float = 90.0,
) -> Dict[str, Any]:
    ok, err = _init_vertex(project, location)
    if not ok:
        raise RuntimeError(err or "Failed to init vertex")

    try:
        from vertexai.generative_models import GenerativeModel, Part  # type: ignore
        model = GenerativeModel("gemini-2.5-flash")

        # Always try to compress to keep inline and reduce token usage
        comp_bytes = _compress_for_inline(video_bytes, content_type or "video/mp4", max_seconds=max_seconds)
        comp_size = len(comp_bytes)
        used_inline = comp_size <= MAX_INLINE_BYTES
        if len(comp_bytes) > MAX_INLINE_BYTES and not gcs_bucket:
            raise RuntimeError(
                f"Video too large for inline upload after compression (~{len(comp_bytes)/1024/1024:.1f} MB). "
                "Trim to <= 90s or reduce resolution/bitrate."
            )

        # Helper to call model with strict, scoped prompts and return (obj, raw_text, err)
        import json as pyjson

        def _escape_unescaped_inner_quotes(text: str) -> str:
            """Escape double quotes that appear inside JSON string values."""
            out: list[str] = []
            in_string = False
            escape = False
            length = len(text)
            idx = 0
            while idx < length:
                ch = text[idx]
                if in_string:
                    if escape:
                        out.append(ch)
                        escape = False
                    elif ch == '\\':
                        out.append(ch)
                        escape = True
                    elif ch == '"':
                        j = idx + 1
                        while j < length and text[j].isspace():
                            j += 1
                        next_char = text[j] if j < length else ''
                        if next_char and next_char not in {',', '}', ']', ':'}:
                            out.append('\\')
                            out.append('"')
                        else:
                            out.append(ch)
                            in_string = False
                    else:
                        out.append(ch)
                else:
                    out.append(ch)
                    if ch == '"':
                        in_string = True
                idx += 1
            return "".join(out)

        def _attempt_json_repair(raw_text: str) -> Optional[Dict[str, Any]]:
            if not raw_text:
                return None
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start < 0 or end <= start:
                return None
            candidate = raw_text[start : end + 1]
            candidate = _escape_unescaped_inner_quotes(candidate)
            candidate = candidate.replace('\r\n', '\n').strip()
            candidate = re.sub(r'"PartialErrors"\s*:\s*\[[\s\S]*$', '', candidate).rstrip()
            if candidate.endswith(','):
                candidate = candidate[:-1].rstrip()
            candidate = re.sub(r"\n\s*(\d+)\s*:", "\n", candidate)
            candidate = re.sub(r"}\s*\n\s*{", "},\n{", candidate)
            candidate = re.sub(r"\]\\s*\\n\\s*{", "],\n{", candidate)
            candidate = re.sub(r"}\s*\n\s*(\")", r"},\n\1", candidate)
            candidate = re.sub(r"\]\\s*\\n\\s*(\")", r"],\n\1", candidate)
            curly_delta = candidate.count('{') - candidate.count('}')
            if curly_delta > 0:
                candidate = candidate + ('}' * curly_delta)
            bracket_delta = candidate.count('[') - candidate.count(']')
            if bracket_delta > 0:
                candidate = candidate + (']' * bracket_delta)
            try:
                return pyjson.loads(candidate)
            except Exception:
                return None

        def call_json(prompt_text: str, retries: int = 2) -> Tuple[Optional[Dict[str, Any]], str, Optional[str]]:
            current_prompt = prompt_text
            raw_text_cache = ""
            last_err: Optional[str] = None
            for attempt in range(retries + 1):
                try:
                    if used_inline:
                        parts = [Part.from_data(mime_type=content_type or "video/mp4", data=comp_bytes), current_prompt]
                    else:
                        parts = [Part.from_data(mime_type=content_type or "video/mp4", data=comp_bytes), current_prompt]
                    resp_local = model.generate_content(
                        parts,
                        generation_config={
                            "temperature": 0.05,
                            "max_output_tokens": 2048,
                            "response_mime_type": "application/json",
                        },
                    )
                    cand = (resp_local.candidates or [None])[0]
                    if not cand:
                        return None, "", "no candidates"
                    parts_out_l = cand.content.parts or []
                    # JSON channels
                    for p in parts_out_l:
                        jv = getattr(p, "json", None) or getattr(p, "json_value", None)
                        if jv:
                            return jv, "", None
                    # Text fallback
                    text_concat_l = "".join([getattr(p, "text", "") or "" for p in parts_out_l]).strip()
                    raw_text_cache = text_concat_l
                    if not text_concat_l:
                        last_err = "empty text"
                    else:
                        s = text_concat_l.find("{")
                        e = text_concat_l.rfind("}")
                        if s >= 0 and e > s:
                            snippet = text_concat_l[s:e + 1]
                            snippet = _escape_unescaped_inner_quotes(snippet)
                            try:
                                return pyjson.loads(snippet), text_concat_l, None
                            except Exception as pe:
                                repaired_local = _attempt_json_repair(snippet)
                                if repaired_local is not None:
                                    return repaired_local, text_concat_l, None
                                last_err = f"parse error: {pe}"
                        else:
                            last_err = "no json braces"
                except Exception as ce:
                    last_err = str(ce)

                if attempt < retries:
                    reminder_parts = [
                        "STRICT JSON REMINDER: Respond with RFC8259-compliant JSON only.",
                        "Use array syntax like [item1, item2]; never prefix entries with indices such as 0: or 1:.",
                        "Always include commas between objects, escape double quotes as \\\" and newlines as \\n.",
                        "Do not emit extra commentary or keys (for example PartialErrors).",
                    ]
                    if last_err:
                        reminder_parts.append(f"Previous error: {last_err}")
                    current_prompt = "\n".join(reminder_parts) + "\n" + prompt_text
                    continue
                break

            repaired = _attempt_json_repair(raw_text_cache)
            if repaired is not None:
                return repaired, raw_text_cache, None
            return None, raw_text_cache, last_err

        # Build three scoped prompts to keep responses small and stable
        base_rules = (
            "Rules: Respond with JSON ONLY (strict RFC8259). Use exactly these key names. Always include all keys; if nothing found, use empty arrays []. "
            "Format arrays with comma-separated elements; never prefix entries with numeric keys like 0:. "
            "Keep arrays small: transcript<=24, visualText<=24, audioGrammar.issues<=8, visualGrammar.issues<=8, visualSpelling.misspellings<=8. "
            "Deduplicate repeated phrases, limit each text value to <=160 chars, escape double quotes as \\\" and replace newlines with \\n. "
            "Do not add commentary, PartialErrors, or extra fields."
        )
        context_lines = []
        if brand_name:
            context_lines.append(f"Brand name: {brand_name}")
        if brand_context:
            context_lines.append(f"Brand context: {brand_context}")
        context_block = ""
        if context_lines:
            context_block = "Brand context:\n" + "\n".join(context_lines) + "\n"

        def _prompt(body: str) -> str:
            return context_block + body if context_block else body

        p_summary = _prompt(
            "Return JSON with keys: description, logoAnalysis{isConsistent,reasoning,identifiedLogo}, textCoherency{score,analysis}, textExtraction{brandMentionCount,nonsenseWords,notableTopics}. "
            "Keep description <= 120 words. textExtraction.brandMentionCount must be an integer count of how often the brand is said. "
            "Limit textExtraction.nonsenseWords<=4 and notableTopics<=4 with short phrases." + base_rules
        )
        p_transcript = _prompt(
            "Return JSON with keys: transcript (array of {startSec,endSec,text}), audioGrammar{issues:[{timeHintSec,message,severity,suggestion}]}. "
            "Provide timestamps; approximate if needed. " + base_rules
        )
        p_visual = _prompt(
            "Return JSON with keys: visualText (array of {startSec?,endSec?,text}), visualGrammar{issues:[{timeHintSec?,message,severity,suggestion?}]}, visualSpelling{misspellings:[{word,timeHintSec?,suggestion?}]}. "
            + base_rules
        )

        # Execute calls
        summary_obj, summary_raw, summary_err = call_json(p_summary, retries=2)
        transcript_obj, transcript_raw, transcript_err = call_json(p_transcript, retries=2)
        visual_obj, visual_raw, visual_err = call_json(p_visual, retries=2)

        # Merge results
        result: Dict[str, Any] = {}
        if isinstance(summary_obj, dict):
            result.update(summary_obj)
        if isinstance(transcript_obj, dict):
            result.update(transcript_obj)
        if isinstance(visual_obj, dict):
            result.update(visual_obj)

        # Clamp oversized lists so downstream JSON stays manageable
        def _limit_list(key: str, limit: int):
            arr = result.get(key)
            if isinstance(arr, list) and len(arr) > limit:
                result[key] = arr[:limit]

        _limit_list('transcript', 24)
        _limit_list('visualText', 24)
        ag = result.get('audioGrammar')
        if isinstance(ag, dict) and isinstance(ag.get('issues'), list) and len(ag['issues']) > 8:
            ag['issues'] = ag['issues'][:8]
        vg = result.get('visualGrammar')
        if isinstance(vg, dict) and isinstance(vg.get('issues'), list) and len(vg['issues']) > 8:
            vg['issues'] = vg['issues'][:8]
        vs = result.get('visualSpelling')
        if isinstance(vs, dict) and isinstance(vs.get('misspellings'), list) and len(vs['misspellings']) > 8:
            vs['misspellings'] = vs['misspellings'][:8]

        # Attach troubleshooting info
        result.setdefault('AnalysisMeta', {})
        result['AnalysisMeta'].update({'compressedBytes': comp_size, 'usedInline': used_inline, 'maxSeconds': max_seconds})
        raw_map = {}
        if summary_raw:
            raw_map['summaryRawText'] = summary_raw
        if transcript_raw:
            raw_map['transcriptRawText'] = transcript_raw
        if visual_raw:
            raw_map['visualRawText'] = visual_raw
        if raw_map:
            result['RawResponses'] = raw_map
        err_list = []
        for label, e in [('summary', summary_err), ('transcript', transcript_err), ('visual', visual_err)]:
            if e:
                err_list.append({ 'stage': label, 'error': e })
        if err_list:
            result['PartialErrors'] = err_list

        return result
    except Exception as e:
        raise
