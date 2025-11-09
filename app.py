import os
# Set matplotlib backend and cache dir to avoid headless/font warnings
os.environ.setdefault('MPLBACKEND', 'Agg')
import tempfile, json, warnings
from typing import Optional
import streamlit as st

# Silence noisy third-party warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*', category=UserWarning)

from cloud_brand_analysis import CloudBrandAnalyzer

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
st.caption('Pacing: P_t = exp(-λ (f_t − f*)²). Goals: hook=2.0 cps, explainer=0.8 cps, calm_brand=0.4 cps.')
