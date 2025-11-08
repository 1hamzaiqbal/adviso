import streamlit as st
import tempfile, os, json

st.set_page_config(page_title='Ad Attention Analyzer', layout='wide')
st.title('Ad Attention Analyzer')

uploaded = st.file_uploader('Upload an MP4', type=['mp4'])
use_clip = st.checkbox('Use CLIP scoring (slower, optional)', value=False)
fps = st.slider('Sampling FPS', 1, 6, 2)

goal = st.selectbox('Creative Goal', ['hook','explainer','calm_brand'], index=0)
age_group = st.selectbox('Target Age Group', 
                         ['general', 'gen_z', 'millennial', 'gen_x', 'boomer', 'children'], 
                         index=0,
                         help='Age group affects scoring weights, thresholds, and pacing preferences')
scene_method = st.selectbox('Scene-change method', ['hist','clip'], index=0)
lam_override = st.text_input('Lambda override (optional)', '')

if uploaded is not None and st.button('Analyze'):
    with tempfile.TemporaryDirectory() as tmpd:
        video_path = os.path.join(tmpd, uploaded.name)
        with open(video_path, 'wb') as f:
            f.write(uploaded.read())

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
        cli_main()

        rep = json.load(open(os.path.join(out_dir, 'report.json')))

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
            st.video(os.path.join(out_dir, 'overlay.mp4'))
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
st.caption('Pacing: P_t = exp(-λ (f_t − f*)²). Goals: hook=2.0 cps, explainer=0.8 cps, calm_brand=0.4 cps.')
