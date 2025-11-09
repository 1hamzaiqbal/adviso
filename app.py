import os
import re
os.environ.setdefault('MPLBACKEND', 'Agg')  # keep matplotlib headless-friendly
import json
import tempfile
import warnings
from typing import Any, Dict, List, Optional

import streamlit as st

from vertex_direct import analyze_brand_vertex

warnings.filterwarnings(
    'ignore',
    message='pkg_resources is deprecated as an API.*',
    category=UserWarning,
)

st.set_page_config(page_title='Ad Attention Analyzer', layout='wide')

st.title('Ad Attention Analyzer')
st.caption('Predict attention, inspect pacing, and optionally run a quick Vertex AI brand check.')


def _run_local_cli(
    video_path: str,
    out_dir: str,
    fps: int,
    goal: str,
    age_group: str,
    scene_method: str,
    use_clip: bool,
    lam_override: Optional[str],
    brand_logo_path: Optional[str],
    brand_colors: Optional[str],
    brand_terms: Optional[str],
    ocr_text: bool,
    logo_threshold: float,
) -> None:
    import sys

    args = [
        'assess_ad',
        '--video',
        video_path,
        '--out',
        out_dir,
        '--fps',
        str(fps),
        '--goal',
        goal,
        '--age-group',
        age_group,
        '--scene-method',
        scene_method,
    ]
    if use_clip:
        args.append('--use-clip')
    if lam_override and lam_override.strip():
        args.extend(['--lambda', lam_override.strip()])
    if brand_logo_path:
        args.extend(['--brand-logo', brand_logo_path])
    if brand_colors:
        args.extend(['--brand-colors', brand_colors])
    if brand_terms:
        args.extend(['--brand-terms', brand_terms])
    if ocr_text:
        args.append('--ocr-text')
    if logo_threshold is not None:
        args.extend(['--logo-threshold', f"{logo_threshold:.2f}"])

    sys.argv = args
    from src.assess_ad import main as cli_main

    cli_main()


def _render_summary_tab(report: Dict[str, Any]) -> None:
    interp = report.get('ScoreInterpretation') or {}
    details = interp.get('detailed_explanation') or {}
    overall = float(interp.get('overall_score', 0.0))
    first5 = float(report.get('First5sRetention', 0.0))
    avg_cut = float(report.get('AvgCutRate', 0.0))

    col_grade, col_overall, col_hook = st.columns(3)
    col_grade.metric('Grade', interp.get('grade', '–'), interp.get('rating', ''))
    col_overall.metric('Overall Attention', f"{overall * 100:.1f}%")
    col_hook.metric('First 5s Retention', f"{first5 * 100:.1f}%")
    st.progress(min(max(overall, 0.0), 1.0))
    st.caption(
        f"Goal: {report.get('Goal', 'hook')} • Age group: {report.get('AgeGroup', 'general')} • "
        f"Avg cuts/sec: {avg_cut:.2f}"
    )

    col_strengths, col_weaknesses = st.columns(2)
    col_strengths.subheader('Strengths')
    for item in details.get('strengths', []) or ['(none noted)']:
        col_strengths.markdown(f"- {item}")
    col_strengths.subheader('Key Insights')
    col_strengths.markdown(f"**Hook:** {details.get('hook_analysis', '—')}")
    col_strengths.markdown(f"**Pacing:** {details.get('pacing_analysis', '—')}")

    col_weaknesses.subheader('Areas to Improve')
    for item in details.get('weaknesses', []) or ['(none noted)']:
        col_weaknesses.markdown(f"- {item}")
    col_weaknesses.subheader('Recommendations')
    for item in details.get('recommendations', []) or ['Keep iterating with creative tests.']:
        col_weaknesses.markdown(f"- {item}")

    json_payload = json.dumps(report, ensure_ascii=False, indent=2)
    st.download_button(
        'Download report.json',
        data=json_payload.encode('utf-8'),
        file_name='report.json',
        mime='application/json',
        use_container_width=True,
    )
    with st.expander('Scorecard JSON'):
        st.json(report)


def _render_visuals_tab(out_dir: str, report: Dict[str, Any]) -> None:
    overlay_path = os.path.join(out_dir, 'overlay.mp4')
    if os.path.exists(overlay_path):
        st.subheader('Heatmap Overlay')
        try:
            st.video(overlay_path)
        except Exception:
            with open(overlay_path, 'rb') as vid_file:
                st.download_button('Download overlay.mp4', vid_file, file_name='overlay.mp4')
    else:
        st.info('Overlay video not produced.')

    cols = st.columns(3)
    plots = [
        ('Attention Curve', 'attention_curve.png', cols[0]),
        ('Editing Rhythm', 'editing_rhythm.png', cols[1]),
        ('Pacing Score', 'pacing_score.png', cols[2]),
    ]
    for title, filename, container in plots:
        img_path = os.path.join(out_dir, filename)
        with container:
            if os.path.exists(img_path):
                container.image(img_path, caption=title, use_column_width=True)
            else:
                container.caption(f'{title}: not available')

    key_moments = report.get('KeyMoments') or []
    if key_moments:
        st.markdown('#### Key Moments')
        st.dataframe(key_moments, hide_index=True, use_container_width=True)


def _coerce_segments(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [seg for seg in value if isinstance(seg, dict)]
    if isinstance(value, dict) and isinstance(value.get('segments'), list):
        return [seg for seg in value['segments'] if isinstance(seg, dict)]
    return []


def _format_segment_range(segment: Dict[str, Any]) -> str:
    start = segment.get('startSec')
    end = segment.get('endSec')
    has_start = isinstance(start, (int, float))
    has_end = isinstance(end, (int, float))
    if has_start and has_end:
        return f"[{start:.1f}s → {end:.1f}s]"
    if has_start:
        return f"[{start:.1f}s]"
    if has_end:
        return f"[→ {end:.1f}s]"
    return ''


SENSITIVE_TERMS = {
    'blood',
    'violence',
    'weapon',
    'injury',
    'explicit',
    'hate',
    'gambling',
    'alcohol',
    'drugs',
}

CTA_TERMS = {
    'sign up',
    'download',
    'buy now',
    'learn more',
    'start free',
    'subscribe',
    'get started',
}


def _evaluate_transcript_heuristics(
    transcript: List[Dict[str, Any]], text_extraction: Dict[str, Any]
) -> Dict[str, Any]:
    combined_text = ' '.join(str(seg.get('text', '')) for seg in transcript).lower()
    sensitive_hits = sorted({term for term in SENSITIVE_TERMS if re.search(rf"\\b{re.escape(term)}\\b", combined_text)})
    cta_hits = sorted({term for term in CTA_TERMS if term in combined_text})

    brand_mentions: List[str] = []
    mention_count: Optional[int] = None
    if isinstance(text_extraction, dict):
        if isinstance(text_extraction.get('brandMentions'), list):
            brand_mentions = [str(item) for item in text_extraction.get('brandMentions', [])][:5]
            mention_count = len(text_extraction.get('brandMentions'))
        else:
            raw_count = text_extraction.get('brandMentionCount')
            if isinstance(raw_count, (int, float)):
                mention_count = int(raw_count)

    return {
        'sensitive_hits': sensitive_hits,
        'cta_hits': cta_hits,
        'brand_mentions': brand_mentions,
        'brand_mention_count': mention_count,
    }


def _render_cloud_tab(cloud_result: Optional[Dict[str, Any]], cloud_err: Optional[str]) -> None:
    if cloud_err:
        st.error(f'Cloud analysis failed: {cloud_err}')
        return
    if not cloud_result:
        st.info('Cloud analysis was skipped.')
        return

    st.subheader('Brand Summary')
    if cloud_result.get('description'):
        st.write(cloud_result['description'])

    cols = st.columns(3)
    logo = (cloud_result.get('logoAnalysis') or {}) if isinstance(cloud_result, dict) else {}
    tc = (cloud_result.get('textCoherency') or {}) if isinstance(cloud_result, dict) else {}
    text_extraction = (cloud_result.get('textExtraction') or {}) if isinstance(cloud_result, dict) else {}

    with cols[0]:
        st.metric('Logo consistent', str(logo.get('isConsistent', 'unknown')))
        if logo.get('identifiedLogo'):
            st.caption(f"Identified: {logo['identifiedLogo']}")
    with cols[1]:
        score = tc.get('score')
        if isinstance(score, (int, float)):
            st.metric('Text coherency', f"{score:.2f}")
        if tc.get('analysis'):
            st.caption(tc['analysis'])
    with cols[2]:
        brand_count = text_extraction.get('brandMentionCount')
        if brand_count is None:
            raw_mentions = text_extraction.get('brandMentions')
            if isinstance(raw_mentions, list):
                brand_count = len(raw_mentions)
            elif isinstance(raw_mentions, (int, float)):
                brand_count = int(raw_mentions)
        st.metric('Brand mentions', str(brand_count if brand_count is not None else 'n/a'))
        extra_bits = []
        nonsense = text_extraction.get('nonsenseWords') or []
        if isinstance(nonsense, list) and nonsense:
            extra_bits.append('Nonsense: ' + ', '.join(nonsense[:4]))
        topics = text_extraction.get('notableTopics') or []
        if isinstance(topics, list) and topics:
            extra_bits.append('Topics: ' + ', '.join(topics[:4]))
        for line in extra_bits:
            st.caption(line)

    transcript = _coerce_segments(cloud_result.get('transcript'))
    if transcript:
        st.markdown('#### Transcript Highlights')
        for seg in transcript[:5]:
            prefix = _format_segment_range(seg)
            text = seg.get('text', '')
            st.write(f"{prefix} {text}".strip())

    visual_text = _coerce_segments(cloud_result.get('visualText'))
    if visual_text:
        st.markdown('#### On-screen Text')
        for seg in visual_text[:5]:
            prefix = _format_segment_range(seg)
            txt = seg.get('text', '')
            st.write(f"{prefix} {txt}".strip())

    heuristics = _evaluate_transcript_heuristics(transcript, text_extraction)
    st.markdown('#### Brand & Safety Heuristics')
    heur_cols = st.columns(3)
    heur_cols[0].metric(
        'CTA coverage',
        'Present' if heuristics['cta_hits'] else 'Needs CTA',
        ', '.join(heuristics['cta_hits']) if heuristics['cta_hits'] else None,
    )
    mention_count = heuristics['brand_mention_count']
    heur_cols[1].metric(
        'Brand mentions',
        str(mention_count if mention_count is not None else 'n/a'),
        ', '.join(heuristics['brand_mentions']) if heuristics['brand_mentions'] else None,
    )
    heur_cols[2].metric(
        'Safety flags',
        'Yes' if heuristics['sensitive_hits'] else 'None noted',
        ', '.join(heuristics['sensitive_hits']) if heuristics['sensitive_hits'] else None,
    )
    if heuristics['sensitive_hits']:
        st.warning(
            'Review transcript for potential safety concerns involving: '
            + ', '.join(heuristics['sensitive_hits'])
        )
    else:
        st.caption('No obvious sensitive language detected in transcript sample.')

    partial_errors = cloud_result.get('PartialErrors') if isinstance(cloud_result, dict) else None
    if partial_errors:
        st.warning(f"Partial errors during Vertex calls: {partial_errors}")

    meta = cloud_result.get('AnalysisMeta') if isinstance(cloud_result, dict) else None
    if meta:
        size_mb = float(meta.get('compressedBytes', 0.0)) / 1_048_576.0
        st.caption(
            f"Vertex payload: {size_mb:.2f} MB, inline={meta.get('usedInline')}, maxSeconds={meta.get('maxSeconds')}"
        )

    with st.expander('Cloud JSON result'):
        st.json(cloud_result)


with st.form('analysis_form'):
    st.subheader('Video & Scoring Settings')
    uploaded = st.file_uploader('Upload MP4', type=['mp4'], help='Limit ~200MB')
    col_opts = st.columns(3)
    with col_opts[0]:
        use_clip = st.checkbox('Use CLIP scoring', value=False)
        fps = st.slider('Sampling FPS', 1, 6, 2)
    with col_opts[1]:
        goal = st.selectbox('Creative goal', ['hook', 'explainer', 'calm_brand'], index=0)
        scene_method = st.selectbox('Scene change method', ['hist', 'clip'], index=0)
    with col_opts[2]:
        age_group = st.selectbox(
            'Target age group',
            ['general', 'gen_z', 'millennial', 'gen_x', 'boomer', 'children'],
            index=0,
        )
        lam_override = st.text_input('Lambda override (optional)', '')

    st.subheader('Cloud Brand Analysis (Vertex, optional)')
    run_cloud = st.checkbox('Enable Vertex check', value=True)
    cloud_cols = st.columns(3)
    project_default = os.getenv('GOOGLE_CLOUD_PROJECT', 'advertigo')
    with cloud_cols[0]:
        project_id = st.text_input('GCP Project', value=project_default)
        max_secs = st.slider('Max seconds to send', 10, 120, 60)
    with cloud_cols[1]:
        vertex_loc = st.text_input('Vertex location', value='us-central1')
        brand_name = st.text_input('Brand / Product name', '')
    with cloud_cols[2]:
        brand_context = st.text_area('Brand context / mission (optional)', height=80)

    submitted = st.form_submit_button('Run analysis', use_container_width=True)


if submitted:
    if not uploaded:
        st.error('Upload an MP4 to analyze.')
    else:
        with tempfile.TemporaryDirectory() as tmpd:
            video_path = os.path.join(tmpd, uploaded.name or 'video.mp4')
            video_bytes = uploaded.getvalue()
            with open(video_path, 'wb') as vid_file:
                vid_file.write(video_bytes)

            out_dir = os.path.join(tmpd, 'out')
            os.makedirs(out_dir, exist_ok=True)

            with st.spinner('Running attention + pacing analysis...'):
                _run_local_cli(
                    video_path=video_path,
                    out_dir=out_dir,
                    fps=fps,
                    goal=goal,
                    age_group=age_group,
                    scene_method=scene_method,
                    use_clip=use_clip,
                    lam_override=lam_override,
                    brand_logo_path=None,
                    brand_colors=None,
                    brand_terms=None,
                    ocr_text=False,
                    logo_threshold=0.6,
                )

            report_path = os.path.join(out_dir, 'report.json')
            report = json.load(open(report_path, 'r', encoding='utf-8'))

            cloud_result: Optional[Dict[str, Any]] = None
            cloud_err: Optional[str] = None
            if run_cloud:
                with st.spinner('Calling Vertex AI...'):
                    try:
                        if not project_id:
                            raise RuntimeError('Provide a GCP project to run the cloud check.')
                        cloud_result = analyze_brand_vertex(
                            video_bytes=video_bytes,
                            filename=uploaded.name or 'video.mp4',
                            content_type=uploaded.type or 'video/mp4',
                            project=project_id,
                            location=vertex_loc or 'us-central1',
                            gcs_bucket=None,
                            max_seconds=float(max_secs),
                            brand_name=brand_name.strip() or None,
                            brand_context=brand_context.strip() or None,
                        )
                    except Exception as exc:  # pragma: no cover - surfaced to user
                        cloud_err = str(exc)

            summary_tab, visuals_tab, cloud_tab = st.tabs(
                ['Score Summary', 'Visual Outputs', 'Cloud Brand']
            )
            with summary_tab:
                _render_summary_tab(report)
            with visuals_tab:
                _render_visuals_tab(out_dir, report)
            with cloud_tab:
                _render_cloud_tab(cloud_result, cloud_err)
else:
    st.info('Upload an MP4 and click "Run analysis" to get started.')
