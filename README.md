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

### Optional: Cloud Brand Coherence (Vertex AI)

This app can call a Node/Express backend (deployed on Cloud Run or run locally) to run a brand coherence analysis using GCS + Vertex AI.

- Set `ADVALUATE_BACKEND_URL` (env var) or paste the URL into the Streamlit field.
- Toggle "Run Cloud Brand Analysis" and optionally enter Brand Name/Mission.
- On Analyze, the app uploads your video via a signed URL and requests analysis. Results render in the UI.

Backend API requirements:
- Endpoints: `GET /api/sign-upload`, `POST /api/analyze`
- Env (backend): `UPLOAD_BUCKET`, `VERTEX_LOCATION` (default `us-central1`), Google Cloud credentials

If you use the prepared Cloud Run deployment, point `ADVALUATE_BACKEND_URL` to your service URL (e.g., `https://advaluate-api-xxxxx-uc.a.run.app`).

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
