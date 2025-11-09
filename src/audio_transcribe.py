import os
import tempfile
from typing import Any, Dict, List, Optional


def _ensure_ffmpeg_in_path() -> None:
    """Ensure an ffmpeg binary is available on PATH using imageio-ffmpeg if needed."""
    from shutil import which
    if which("ffmpeg") is not None:
        return
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
        ffmpeg_path = get_ffmpeg_exe()
        bin_dir = os.path.dirname(ffmpeg_path)
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        # If we can't ensure, whisper/moviepy may still work if system ffmpeg exists
        pass


def transcribe_audio(video_path: str, max_seconds: Optional[float] = 90.0, model_name: str = "base") -> Dict[str, Any]:
    """
    Transcribe the video's audio locally using OpenAI Whisper (CPU by default).
    Returns: { available: bool, segments: [{startSec, endSec, text}], reason? }
    """
    try:
        _ensure_ffmpeg_in_path()
        # Extract an audio subclip to WAV using moviepy (falls back to imageio-ffmpeg binary)
        from moviepy.editor import VideoFileClip  # type: ignore
        with VideoFileClip(video_path) as clip:
            if max_seconds is not None:
                end = min(max_seconds, clip.duration)
                if end < clip.duration:
                    clip = clip.subclip(0, end)
            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "audio.wav")
                # write_audiofile may be noisy; suppress logs
                clip.audio.write_audiofile(wav_path, verbose=False, logger=None)

                # Load whisper model lazily
                try:
                    import whisper  # type: ignore
                except Exception as e:
                    return { 'available': False, 'reason': f'whisper not installed: {e}', 'segments': [] }

                try:
                    model = whisper.load_model(model_name)
                except Exception:
                    # Fallback to small if base missing
                    model = whisper.load_model("small")

                res = model.transcribe(wav_path, language="en")
                segs: List[Dict[str, Any]] = []
                for s in res.get('segments', []) or []:
                    try:
                        segs.append({
                            'startSec': float(s.get('start', 0.0)),
                            'endSec': float(s.get('end', 0.0)),
                            'text': str(s.get('text', '')).strip()
                        })
                    except Exception:
                        continue
                return { 'available': True, 'segments': segs }
    except Exception as e:
        return { 'available': False, 'reason': str(e), 'segments': [] }

