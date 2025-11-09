# Ad Attention Analyzer (MVP)

Predict and visualize whether an ad **captures attention** in the first 3–5 seconds.

**What you get**
- **Attention heatmaps** (per-frame) using OpenCV Spectral Residual saliency
- **Motion energy** (optical flow) for early dynamics
- **Optional CLIP score** comparing frames to prompts (“eye‑catching ad” vs “boring ad”)
- **Attention curve** over time + JSON scorecard
- **Overlay video** with heatmap blended onto your ad
- **Streamlit app** for interactive uploads and comparisons

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> CLIP is optional (improves estimates). If CUDA not available, it will run on CPU.

## Quick start (CLI)

```bash
python -m src.assess_ad \\
  --video sample.mp4 \\
  --out ./out \\
  --fps 2 \\
  --use-clip
```

Outputs in `./out`:
- `report.json` – scores + key moments
- `attention_curve.png` – time series
- `overlay.mp4` – heatmap blended video

## Streamlit UI

```bash
streamlit run app.py
```

Upload an MP4 to see heatmaps, curve, and the JSON report.

### Cloud Brand Coherence (Vertex Direct)

The app now calls Vertex AI directly from Python — no separate backend required.

- Set credentials via Application Default Credentials:
  - `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json`
- Provide your GCP Project and Vertex Location in the UI.
- On Analyze, the app compresses the video (up to the selected duration) to fit an inline Vertex request; results render in the UI.

Notes:
- For long videos, the app trims to your chosen max seconds and downscales to keep the request inline (~15MB limit).
- Returned JSON includes description, logoAnalysis, textCoherency, textExtraction, transcript, audioGrammar, visualText, visualGrammar, and visualSpelling; the UI renders these.

### Visual + Audio Word/Grammar Checkers

The UI includes two complementary checkers:

- Visual OCR + Grammar (on-screen text)
  - Uses EasyOCR to extract visible text at ~1 fps; timestamps reflect frame times.
  - Runs spelling via `pyspellchecker` and optional grammar via `language_tool_python`.
  - If dependencies are missing, the app will show a notice and skip this step.

- Audio Transcript + Grammar (cloud)
  - Uses the Cloud backend (Vertex) to produce a transcript with approximate timestamps and grammar issues.
  - Displayed when Cloud Brand Analysis is enabled.

Install extras (if you didn’t install via requirements yet):

```bash
pip install easyocr pyspellchecker language_tool_python
```

Notes:
- EasyOCR uses PyTorch under the hood; Apple Silicon is supported via arm64 wheels.
- `language_tool_python` may launch a local Java server if available; otherwise it may fall back. If grammar is not critical for your workflow, you can skip installing it.

## Theory (short)
We approximate attention using:
- **Saliency (SpectralResidual)** → where eyes are likely drawn in each frame.
- **Motion energy** → higher early motion tends to increase hook rate.
- **Text clarity proxy** → CLIP similarity to “product / CTA” prompts (optional).

Final score combines normalized saliency concentration, early motion, and CLIP deltas.
Weights are configurable in `src/attention_score.py`.

## License
MIT (for this scaffold). Ensure your input content complies with platform policies.
# addddddd
