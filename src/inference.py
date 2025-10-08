import torch, numpy as np, os
from .model import SimpleCNN
from .utils import audio_to_mel
from .config import GENRES

def load_model(path='best_model.pth', device=None):
    device = device or torch.device('cpu')
    model = SimpleCNN(num_classes=len(GENRES))
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    return model

def predict_file(model, file_path, device=None):
    device = device or torch.device('cpu')
    S = audio_to_mel(file_path)
    x = np.expand_dims(S, axis=(0,1))
    import torch as _torch
    xb = _torch.tensor(x, dtype=_torch.float).to(device)
    with _torch.no_grad():
        logits = model(xb)
        probs = _torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
    return {'genre': GENRES[pred_idx], 'probs': probs.tolist()}
