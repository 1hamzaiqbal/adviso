import torch, numpy as np
from PIL import Image

# Simple module-level cache so we don't reload CLIP each run
_CLIP_CACHE = {
    'model': None,
    'preprocess': None,
    'device': 'cpu'
}

def try_load_clip(device='cpu'):
    # Return cached if available
    if _CLIP_CACHE['model'] is not None and _CLIP_CACHE['preprocess'] is not None:
        return _CLIP_CACHE['model'], _CLIP_CACHE['preprocess'], _CLIP_CACHE['device']
    try:
        import clip
        model, preprocess = clip.load('ViT-B/32', device=device, download=True)
        _CLIP_CACHE['model'] = model
        _CLIP_CACHE['preprocess'] = preprocess
        _CLIP_CACHE['device'] = device
        return model, preprocess, device
    except Exception:
        return None, None, device

def score_frames_with_prompts(frames_bgr, prompts=('an eye-catching ad','a boring ad'), device='cpu'):
    model, preprocess, device = try_load_clip(device)
    if model is None:
        return [0.5]*len(frames_bgr), 0.0

    import clip
    text = clip.tokenize(list(prompts)).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    scores = []
    for bgr in frames_bgr:
        img = Image.fromarray(bgr[:,:,::-1])
        inp = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(inp)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).softmax(dim=-1).squeeze(0).cpu().numpy()
            scores.append(float(sim[0]))
    delta = float(np.mean(scores) - 0.5)
    return scores, delta
