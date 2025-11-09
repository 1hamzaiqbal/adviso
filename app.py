import os
# Set matplotlib backend and cache dir to avoid headless/font warnings
os.environ.setdefault('MPLBACKEND', 'Agg')
import tempfile, json, warnings
from typing import Optional
import streamlit as st

# Silence noisy third-party warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*', category=UserWarning)

# Direct Vertex analysis (no backend)
from vertex_direct import analyze_brand_vertex

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

st.markdown('---')
st.subheader('Cloud Analysis (Vertex direct)')
st.caption('Calls Vertex AI directly from this app — no buckets required for short videos. Longer videos are compressed inline.')
brand_name: str = st.text_input('Brand Name (optional)', '')
brand_mission: str = st.text_area('Brand Mission (optional)', '')
project_id = st.text_input('GCP Project', value='advertigo')
vertex_loc = st.text_input('Vertex Location', value='us-central1')
max_secs = st.slider('Max seconds to analyze (compressed)', 10, 120, 60, help='Lower this if cloud parsing fails; shorter clips are more reliable.')

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
        with st.spinner('Analyzing with Vertex (direct)...'):
            try:
                if not project_id:
                    raise RuntimeError('Set GCP Project in the UI or GOOGLE_CLOUD_PROJECT env var')
                cloud_result = analyze_brand_vertex(
                    video_bytes=video_bytes,
                    filename=uploaded.name or 'video.mp4',
                    content_type=uploaded.type or 'video/mp4',
                    project=project_id,
                    location=vertex_loc or 'us-central1',
                    gcs_bucket=None,
                    max_seconds=float(max_secs),
                )
            except Exception as e:
                cloud_err = str(e)

        # No local OCR/transcription in cloud-direct mode

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
            try:
                # Try to fill missing summary fields from RawResponses.summaryRawText
                def _json_from_text(raw_text: str):
                    try:
                        import json as _pyjson
                        s = raw_text.find('{'); e = raw_text.rfind('}')
                        if s >= 0 and e > s:
                            return _pyjson.loads(raw_text[s:e+1])
                    except Exception:
                        return None
                    return None

                try:
                    raws_all = (cloud_result.get('RawResponses') if isinstance(cloud_result, dict) else None) or {}
                    sraw = raws_all.get('summaryRawText')
                    sjson = _json_from_text(sraw) if sraw else None
                    if isinstance(sjson, dict):
                        for k in ('description','logoAnalysis','textCoherency','textExtraction'):
                            v = cloud_result.get(k)
                            if not v or (isinstance(v, dict) and not v):
                                cloud_result[k] = sjson.get(k)
                except Exception:
                    pass

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
                # Text extraction + brand mentions (recomputed from transcript/visual)
                te = cloud_result.get('textExtraction') or {}
                st.markdown('#### Text Extraction')
                st.json(te)

                # Compute brand mentions from audio + visual using provided brand_name
                def _collect_mentions(name: str, items, ts_key_start='startSec', ts_key_end='endSec', txt_key='text'):
                    out = []
                    if not name or not items:
                        return out
                    name_l = name.strip().lower()
                    for seg in items:
                        try:
                            txt = str(seg.get(txt_key, '')).lower()
                            if not txt:
                                continue
                            if name_l in txt:
                                out.append({
                                    'startSec': float(seg.get(ts_key_start, 0.0)) if isinstance(seg.get(ts_key_start), (int,float)) else None,
                                    'endSec': float(seg.get(ts_key_end, 0.0)) if isinstance(seg.get(ts_key_end), (int,float)) else None,
                                    'text': seg.get(txt_key, '')
                                })
                        except Exception:
                            continue
                    return out

                # Helpers to extract segments with flexible shapes/keys
                def _parse_segments_from_value(val):
                    if isinstance(val, list):
                        return val
                    if isinstance(val, dict):
                        segs = val.get('segments')
                        if isinstance(segs, list):
                            return segs
                    return []

                def _json_from_text(raw_text: str):
                    try:
                        import json as _pyjson
                        s = raw_text.find('{')
                        e = raw_text.rfind('}')
                        if s >= 0 and e > s:
                            return _pyjson.loads(raw_text[s:e+1])
                    except Exception:
                        return None
                    return None

                # Parse transcript segments of either shape with fallbacks
                tr_obj = cloud_result.get('transcript') if isinstance(cloud_result, dict) else None
                tr_segs = _parse_segments_from_value(tr_obj)
                if not tr_segs:
                    # Fallback: parse from raw text if available
                    raws = (cloud_result.get('RawResponses') if isinstance(cloud_result, dict) else None) or {}
                    tr_raw = raws.get('transcriptRawText')
                    j = _json_from_text(tr_raw) if tr_raw else None
                    if isinstance(j, dict):
                        tr_segs = _parse_segments_from_value(j.get('transcript'))

                # Parse visual segments with fallbacks
                vt_obj = cloud_result.get('visualText') if isinstance(cloud_result, dict) else None
                vt_segs = _parse_segments_from_value(vt_obj)
                if not vt_segs:
                    raws = (cloud_result.get('RawResponses') if isinstance(cloud_result, dict) else None) or {}
                    vt_raw = raws.get('visualRawText')
                    j2 = _json_from_text(vt_raw) if vt_raw else None
                    if isinstance(j2, dict):
                        vt_segs = _parse_segments_from_value(j2.get('visualText'))

                audio_mentions = _collect_mentions(brand_name, tr_segs)
                visual_mentions = _collect_mentions(brand_name, vt_segs)

                # Show brand mentions summary
                st.markdown('#### Brand Mentions (Computed)')
                cma, cmv = st.columns(2)
                with cma:
                    st.markdown(f"Audio: {len(audio_mentions)}")
                    if audio_mentions:
                        for m in audio_mentions:
                            t0 = m.get('startSec'); t1 = m.get('endSec'); txt = m.get('text','')
                            if isinstance(t0, (int,float)) and isinstance(t1, (int,float)):
                                st.caption(f"[{t0:.1f}s → {t1:.1f}s] {txt}")
                            elif isinstance(t0, (int,float)):
                                st.caption(f"t≈{t0:.1f}s: {txt}")
                            else:
                                st.caption(txt)
                with cmv:
                    st.markdown(f"Visual: {len(visual_mentions)}")
                    if visual_mentions:
                        for m in visual_mentions:
                            t0 = m.get('startSec'); t1 = m.get('endSec'); txt = m.get('text','')
                            if isinstance(t0, (int,float)) and isinstance(t1, (int,float)):
                                st.caption(f"[{t0:.1f}s → {t1:.1f}s] {txt}")
                            elif isinstance(t0, (int,float)):
                                st.caption(f"t≈{t0:.1f}s: {txt}")
                            else:
                                st.caption(txt)

                # If cloud textExtraction has brandMentions, show a quick comparison
                try:
                    cloud_bm = te.get('brandMentions')
                    if isinstance(cloud_bm, (int,float)):
                        st.caption(f"Cloud brandMentions: {int(cloud_bm)} • Computed total: {len(audio_mentions)+len(visual_mentions)}")
                except Exception:
                    pass
                with st.expander('Raw Cloud Response'):
                    meta = (cloud_result.get('AnalysisMeta') if isinstance(cloud_result, dict) else None) or {}
                    if meta:
                        size_mb = float(meta.get('compressedBytes', 0))/1024/1024 if meta.get('compressedBytes') else 0
                        st.caption(f"Compressed size sent inline: {size_mb:.2f} MB; usedInline={bool(meta.get('usedInline'))}; maxSeconds={meta.get('maxSeconds')}")
                    # Show any raw texts from multi-call flow
                    raws = (cloud_result.get('RawResponses') if isinstance(cloud_result, dict) else None) or {}
                    for label, raw_val in raws.items():
                        if raw_val:
                            st.markdown(f"{label}:")
                            st.code(raw_val, language='json')
                    # Legacy single-call rawText support
                    if isinstance(cloud_result, dict) and cloud_result.get('rawText'):
                        st.markdown('rawText:')
                        st.code(cloud_result.get('rawText') or '', language='json')
                    # Always provide final merged JSON for debugging
                    st.markdown('Merged JSON:')
                    st.json(cloud_result)
            except Exception:
                st.json(cloud_result)

        st.markdown('---')
        st.subheader('Visual OCR + Grammar (On-screen text)')
        # Cloud outputs (with fallback into raw if needed)
        vt = (cloud_result or {}).get('visualText') if cloud_result else None
        # Accept both shapes: list or {segments:[...]}
        cloud_v_segments = []
        if isinstance(vt, list):
            cloud_v_segments = vt
        elif isinstance(vt, dict):
            cloud_v_segments = vt.get('segments', []) or []
        if not cloud_v_segments and isinstance(cloud_result, dict):
            raws = cloud_result.get('RawResponses') or {}
            vraw = raws.get('visualRawText')
            if vraw:
                try:
                    import json as _pyjson
                    s = vraw.find('{'); e = vraw.rfind('}')
                    if s >= 0 and e > s:
                        j = _pyjson.loads(vraw[s:e+1])
                        val = j.get('visualText')
                        if isinstance(val, list):
                            cloud_v_segments = val
                        elif isinstance(val, dict) and isinstance(val.get('segments'), list):
                            cloud_v_segments = val.get('segments')
                except Exception:
                    pass
        if cloud_v_segments:
            st.markdown('#### Extracted Phrases (cloud)')
            for s in cloud_v_segments:
                try:
                    start = s.get('startSec'); end = s.get('endSec'); txt = s.get('text','')
                    if isinstance(start, (int,float)) and isinstance(end, (int,float)):
                        st.write(f"[{start:.1f}s → {end:.1f}s] {txt}")
                    elif isinstance(start, (int,float)):
                        st.write(f"t≈{start:.1f}s: {txt}")
                    else:
                        st.write(txt)
                except Exception:
                    st.write(s)
            vg = (cloud_result or {}).get('visualGrammar') or {}
            gi = vg.get('issues', []) if isinstance(vg, dict) else []
            if gi:
                st.markdown('#### Visual Grammar Issues (cloud)')
                for m in gi:
                    hint = m.get('timeHintSec')
                    prefix = f"t≈{hint:.1f}s: " if isinstance(hint, (int, float)) else ''
                    sev = m.get('severity','').lower()
                    sev_prefix = f"({sev}) " if sev else ''
                    st.write(f"{prefix}{sev_prefix}{m.get('message','')}")
                    if m.get('suggestion'):
                        st.caption(f"Suggestion: {m['suggestion']}")
            vs = (cloud_result or {}).get('visualSpelling') or {}
            miss = vs.get('misspellings', []) if isinstance(vs, dict) else []
            if miss:
                st.markdown('#### Visual Spelling (cloud)')
                st.json(miss)
        else:
            st.info('No on-screen text detected by cloud.')

        # Audio transcript + grammar
        if cloud_result is not None:
            st.markdown('---')
            st.subheader('Audio Transcript + Grammar (Cloud)')
            tr = (cloud_result or {}).get('transcript', {})
            # Accept both shapes: list or {segments:[...]}
            if isinstance(tr, list):
                segs = tr
            elif isinstance(tr, dict):
                segs = tr.get('segments', [])
            else:
                segs = []
            if segs:
                st.markdown('#### Transcript Segments')
                for s in segs:
                    try:
                        st.write(f"[{float(s.get('startSec',0)):.1f}s → {float(s.get('endSec',0)):.1f}s] {s.get('text','')}")
                    except Exception:
                        st.write(s)
            ag = (cloud_result or {}).get('audioGrammar', {})
            issues = ag.get('issues', []) if isinstance(ag, dict) else []
            # Fallback to raw transcript stage if needed
            if not issues and isinstance(cloud_result, dict):
                raws = cloud_result.get('RawResponses') or {}
                tr_raw = raws.get('transcriptRawText')
                if tr_raw:
                    j = _json_from_text(tr_raw)
                    if isinstance(j, dict):
                        ag2 = j.get('audioGrammar') or {}
                        if isinstance(ag2, dict) and isinstance(ag2.get('issues'), list):
                            issues = ag2.get('issues')
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
        # No local transcript path in cloud-direct mode

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
