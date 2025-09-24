import torch
from .models import ConvAutoencoder, ClassifierFromLatent
from .features import load_audio, compute_mel
from .config import DEVICE
import numpy as np


def predict(audio_path, ae_ckpt, clf_ckpt, labels_map):
    ae = ConvAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(ae_ckpt, map_location=DEVICE))
    ae.eval()
    clf = ClassifierFromLatent(ae.latent_dim, len(labels_map)).to(DEVICE)
    clf.load_state_dict(torch.load(clf_ckpt, map_location=DEVICE))
    clf.eval()

    y = load_audio(audio_path)
    mel = compute_mel(y)
    mel = np.expand_dims(mel, axis=(0,1))
    x = torch.tensor(mel).to(DEVICE)
    with torch.no_grad():
        _, z = ae(x)
        logits = clf(z)
        pred = logits.argmax(1).item()
    inv_map = {v:k for k,v in labels_map.items()}
    return inv_map[pred]
