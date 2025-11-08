import streamlit as st
import tempfile, os, json
from pathlib import Path
from src.assess_ad import main as assess_main

st.set_page_config(page_title='Ad Attention Analyzer', layout='wide')
st.title('Ad Attention Analyzer (MVP)')

uploaded = st.file_uploader('Upload an MP4', type=['mp4'])
use_clip = st.checkbox('Use CLIP scoring (slower, optional)', value=False)
fps = st.slider('Sampling FPS', 1, 6, 2)

if uploaded is not None and st.button('Analyze'):
    with tempfile.TemporaryDirectory() as tmpd:
        video_path = os.path.join(tmpd, uploaded.name)
        with open(video_path, 'wb') as f:
            f.write(uploaded.read())

        out_dir = os.path.join(tmpd, 'out')
        import sys
        sys.argv = ['assess_ad', '--video', video_path, '--out', out_dir, '--fps', str(fps)] + (['--use-clip'] if use_clip else [])
        from src.assess_ad import main as cli_main
        cli_main()

        rep = json.load(open(os.path.join(out_dir, 'report.json')))
        st.subheader('Scorecard')
        st.json(rep)

        st.subheader('Attention Curve')
        st.image(os.path.join(out_dir, 'attention_curve.png'))

        st.subheader('Heatmap Overlay')
        st.video(os.path.join(out_dir, 'overlay.mp4'))
