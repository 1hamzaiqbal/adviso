import os
# Set matplotlib backend and cache dir to avoid headless/font warnings
os.environ.setdefault('MPLBACKEND', 'Agg')
import tempfile, json, warnings
from typing import Optional
import streamlit as st

# Silence noisy third-party warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*', category=UserWarning)

from cloud_brand_analysis import CloudBrandAnalyzer
from src.visual_text import extract_visual_text

st.set_page_config(page_title='Ad Attention Analyzer', layout='wide')
st.title('Ad Attention Analyzer')

uploaded = st.file_uploader('Upload an MP4', type=['mp4'], help='Limit ~200MB; larger files may be slow to process')
use_clip = st.checkbox('Use CLIP scoring (slower, optional)', value=False)
fps = st.slider('Sampling FPS', 1, 6, 2)

goal = st.selectbox('Creative Goal', ['hook','explainer','calm_brand'], index=0)
age_group = st.selectbox('Target Age Group', 
                         ['general', 'gen_z', 'millennial', 'gen_x', 'boomer', 'children'], 
                         index=0,
                         help='Age group affects scoring weights, thresholds, and pacing preferences')
scene_method = st.selectbox('Scene-change method', ['hist','clip'], index=0)
lam_override = st.text_input('Lambda override (optional)', '')

# Cloud brand analysis (from advaluate_build_1 backend)
st.markdown('---')
st.subheader('Cloud Brand Coherence Analysis (Vertex AI)')
st.caption('Optional: uses the advaluate_build_1 backend with GCS + Vertex AI. Configure backend URL to enable.')

default_backend = os.getenv('ADVALUATE_BACKEND_URL', '').strip()
backend_url: str = st.text_input('Backend URL', value=default_backend, help='e.g., https://your-backend.example.com')
brand_name: str = st.text_input('Brand Name (optional)', '')
brand_mission: str = st.text_area('Brand Mission (optional)', '')
run_cloud = st.checkbox('Run Cloud Brand Analysis', value=bool(backend_url))
run_visual_text = st.checkbox('Run Visual OCR + Grammar (beta)', value=True,
                              help='Extract on-screen text with OCR, spellcheck and simple grammar; includes timestamps.')

if uploaded is not None and st.button('Analyze'):
    with tempfile.TemporaryDirectory() as tmpd:
        video_path = os.path.join(tmpd, uploaded.name)
        video_bytes: Optional[bytes] = uploaded.read()
        with open(video_path, 'wb') as f:
            f.write(video_bytes)

        out_dir = os.path.join(tmpd, 'out')
        os.makedirs(out_dir, exist_ok=True)

        import sys
        sys.argv = ['assess_ad',
                    '--video', video_path,
                    '--out', out_dir,
                    '--fps', str(fps),
                    '--goal', goal,
                    '--age-group', age_group,
                    '--scene-method', scene_method] +                     (['--use-clip'] if use_clip else []) +                     (['--lambda', lam_override] if lam_override.strip() != '' else [])

        from src.assess_ad import main as cli_main
        with st.spinner('Running attention + pacing analysis...'):
            cli_main()

        rep = json.load(open(os.path.join(out_dir, 'report.json')))

        cloud_result = None
        cloud_err = None
        if run_cloud and backend_url.strip():
            with st.spinner('Uploading to cloud and analyzing brand coherence...'):
                try:
                    analyzer = CloudBrandAnalyzer(base_url=backend_url.strip())
                    cloud_result = analyzer.run(
                        video_bytes=video_bytes,
                        filename=uploaded.name or 'video.mp4',
                        content_type=uploaded.type or 'video/mp4',
                        brand_name=(brand_name or None),
                        brand_mission=(brand_mission or None)
                    )
                except Exception as e:
                    cloud_err = str(e)

        visual_result = None
        if run_visual_text:
            with st.spinner('Extracting visual text (OCR) and checking grammar...'):
                try:
                    visual_result = extract_visual_text(video_path, fps=1.0, max_frames=90)
                except Exception as e:
                    visual_result = {'available': False, 'reason': str(e), 'segments': []}

        # Display final score prominently
        if 'ScoreInterpretation' in rep:
            interp = rep['ScoreInterpretation']
            
            # Main score display with grade
            st.markdown('---')
            col_score1, col_score2, col_score3 = st.columns([1, 2, 1])
            with col_score2:
                age_info = f" ({rep.get('AgeGroup', 'general').upper()})" if 'AgeGroup' in rep else ""
                st.markdown(f"## 🎯 Final Score: **{interp['grade']}** ({interp['rating']}){age_info}")
                score_pct = interp['overall_score'] * 100
                st.markdown(f"### {score_pct:.1f}%")
                st.progress(interp['overall_score'])
            
            st.markdown('---')
            
            # Performance prediction
            st.markdown(f"### 📊 Performance Prediction")
            st.info(interp['performance_prediction'])
            
            # Detailed explanation
            details = interp['detailed_explanation']
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown("#### ✅ Strengths")
                for strength in details['strengths']:
                    st.markdown(f"- {strength}")
                
                st.markdown("#### 📈 Key Insights")
                st.markdown(f"**Hook Analysis:** {details['hook_analysis']}")
                st.markdown(f"**Pacing Analysis:** {details['pacing_analysis']}")
            
            with col_exp2:
                st.markdown("#### ⚠️ Areas for Improvement")
                for weakness in details['weaknesses']:
                    st.markdown(f"- {weakness}")
                
                st.markdown("#### 💡 Recommendations")
                for rec in details['recommendations']:
                    st.markdown(f"- {rec}")
            
            st.markdown('---')

        c1, c2 = st.columns([3,2])
        with c1:
            st.subheader('Heatmap Overlay')
            ovp = os.path.join(out_dir, 'overlay.mp4')
            try:
                st.video(ovp)
            except Exception:
                st.warning('Overlay video could not be displayed. Download instead:')
                with open(ovp, 'rb') as f:
                    st.download_button('Download overlay.mp4', f, file_name='overlay.mp4', mime='video/mp4')
        with c2:
            st.subheader('Detailed Scorecard')
            st.json(rep)

        st.subheader('Attention Curve')
        st.image(os.path.join(out_dir, 'attention_curve.png'))

        col3, col4 = st.columns(2)
        with col3:
            st.subheader('Editing Rhythm (cuts/sec)')
            st.image(os.path.join(out_dir, 'editing_rhythm.png'))
        with col4:
            st.subheader('Goal-Adjusted Pacing Score')
            st.image(os.path.join(out_dir, 'pacing_score.png'))

        st.markdown('---')
        st.subheader('Cloud Brand Coherence Result')
        if cloud_err:
            st.error(f'Cloud analysis failed: {cloud_err}')
        elif cloud_result is not None:
            # Present a tidy summary similar to advaluate_build_1/types
            try:
                st.markdown('### Brand Summary')
                desc = cloud_result.get('description')
                if desc:
                    st.write(desc)

                colb1, colb2 = st.columns(2)
                with colb1:
                    st.markdown('#### Logo Analysis')
                    la = cloud_result.get('logoAnalysis') or {}
                    st.write({
                        'isConsistent': la.get('isConsistent'),
                        'identifiedLogo': la.get('identifiedLogo'),
                    })
                    if la.get('reasoning'):
                        st.caption(la['reasoning'])
                with colb2:
                    st.markdown('#### Text Coherency')
                    tc = cloud_result.get('textCoherency') or {}
                    score = tc.get('score')
                    if isinstance(score, (int, float)):
                        st.metric('Coherency Score', f"{score}")
                    analysis = tc.get('analysis')
                    if analysis:
                        st.caption(analysis)

                st.markdown('#### Text Extraction')
                te = cloud_result.get('textExtraction') or {}
                st.json(te)

                with st.expander('Raw Cloud Response'):
                    st.json(cloud_result)
            except Exception:
                # Fallback to raw JSON if schema differs
                st.json(cloud_result)

        st.markdown('---')
        st.subheader('Visual OCR + Grammar (On-screen text)')
        if visual_result is None:
            st.info('OCR not requested.')
        elif not visual_result.get('available', False):
            st.warning(f"OCR/Grammar not available: {visual_result.get('reason','unknown')}")
        else:
            segs = visual_result.get('segments', [])
            if segs:
                st.markdown('#### Extracted Phrases (with timestamps)')
                for seg in segs:
                    st.write(f"t={seg['time']:.1f}s: {seg['text']}")
                    if seg.get('misspellings'):
                        st.caption(f"Misspellings: {', '.join(seg['misspellings'])}")
                    if seg.get('grammar_issues'):
                        for gi in seg['grammar_issues']:
                            st.caption(f"Grammar: {gi.get('message','')} → {gi.get('suggestion','')}")
            else:
                st.info('No on-screen text detected.')
            st.markdown('#### Visual Spelling Summary')
            st.json(visual_result.get('misspellings_summary', {}))
            if visual_result.get('grammar_issues_summary'):
                with st.expander('Visual Grammar Issues (detailed)'):
                    st.json(visual_result['grammar_issues_summary'])

        # Audio transcript + grammar (from cloud)
        if cloud_result is not None:
            st.markdown('---')
            st.subheader('Audio Transcript + Grammar (Cloud)')
            tr = (cloud_result or {}).get('transcript', {})
            segs = tr.get('segments', []) if isinstance(tr, dict) else []
            if segs:
                st.markdown('#### Transcript Segments')
                for s in segs:
                    try:
                        st.write(f"[{float(s.get('startSec',0)):.1f}s → {float(s.get('endSec',0)):.1f}s] {s.get('text','')}")
                    except Exception:
                        st.write(s)
            ag = (cloud_result or {}).get('audioGrammar', {})
            issues = ag.get('issues', []) if isinstance(ag, dict) else []
            if issues:
                st.markdown('#### Audio Grammar Issues')
                for m in issues:
                    hint = m.get('timeHintSec')
                    prefix = f"t≈{hint:.1f}s: " if isinstance(hint, (int, float)) else ''
                    sev = m.get('severity','').lower()
                    sev_prefix = f"({sev}) " if sev else ''
                    st.write(f"{prefix}{sev_prefix}{m.get('message','')}")
                    if m.get('suggestion'):
                        st.caption(f"Suggestion: {m['suggestion']}")

st.markdown('---')
st.caption('Pacing: P_t = exp(-λ (f_t − f*)²). Goals: hook=2.0 cps, explainer=0.8 cps, calm_brand=0.4 cps.')
with st.expander('Environment Status', expanded=False):
    def _has(mod):
        try:
            __import__(mod)
            return True
        except Exception:
            return False
    st.write({
        'torch': _has('torch'),
        'clip-anytorch': _has('clip'),
        'easyocr': _has('easyocr'),
        'pyspellchecker': _has('spellchecker'),
        'language_tool_python': _has('language_tool_python')
    })
