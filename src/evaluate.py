import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from .data import make_dataloaders
from .models import ConvAutoencoder, ClassifierFromLatent
from .config import DEVICE
import numpy as np


def evaluate(ae_ckpt, clf_ckpt):
    train_loader, val_loader, labels_map = make_dataloaders()
    n_classes = len(labels_map)
    ae = ConvAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(ae_ckpt, map_location=DEVICE))
    ae.eval()
    clf = ClassifierFromLatent(ae.latent_dim, n_classes).to(DEVICE)
    clf.load_state_dict(torch.load(clf_ckpt, map_location=DEVICE))
    clf.eval()

    ys = []
    ypred = []
    for batch in val_loader:
        x = batch['mel'].to(DEVICE)
        y = batch['label'].numpy().tolist()
        with torch.no_grad():
            _, z = ae(x)
            preds = clf(z).argmax(1).cpu().numpy().tolist()
        ys.extend(y)
        ypred.extend(preds)

    acc = accuracy_score(ys, ypred)
    cm = confusion_matrix(ys, ypred)
    return acc, cm

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae', required=True)
    parser.add_argument('--clf', required=True)
    args = parser.parse_args()
    acc, cm = evaluate(args.ae, args.clf)
    print('Val acc:', acc)
    print('Confusion matrix:\n', cm)
