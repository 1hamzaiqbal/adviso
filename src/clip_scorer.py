import torch, numpy as np
from PIL import Image

def try_load_clip(device='cpu'):
    try:
        import clip
        model, preprocess = clip.load('ViT-B/32', device=device, download=True)
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
