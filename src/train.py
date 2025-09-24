import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from .config import DEVICE, BATCH_SIZE, SEED
from .data import make_dataloaders
from .models import ConvAutoencoder, ClassifierFromLatent
import random
import numpy as np

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ae(epochs=20, latent_dim=128, save_path='checkpoints/ae.pth'):
    set_seed()
    train_loader, val_loader, labels_map = make_dataloaders(batch_size=BATCH_SIZE)
    model = ConvAutoencoder(latent_dim=latent_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"AE Train epoch {epoch+1}/{epochs}"):
            x = batch['mel'].to(DEVICE)
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1} train loss: {train_loss:.6f}")
        torch.save(model.state_dict(), save_path)

    return model, labels_map


def train_classifier(epochs=15, ae_checkpoint='checkpoints/ae.pth', save_path='checkpoints/classifier.pth'):
    set_seed()
    train_loader, val_loader, labels_map = make_dataloaders(batch_size=BATCH_SIZE)
    n_classes = len(labels_map)
    ae = ConvAutoencoder().to(DEVICE)
    ae.load_state_dict(torch.load(ae_checkpoint, map_location=DEVICE))
    ae.eval()
    latent_dim = ae.latent_dim
    clf = ClassifierFromLatent(latent_dim, n_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(epochs):
        clf.train()
        total = 0
        correct = 0
        for batch in tqdm(train_loader, desc=f"Clf Train epoch {epoch+1}/{epochs}"):
            x = batch['mel'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            with torch.no_grad():
                _, z = ae(x)
            preds = clf(z)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += labels.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
        acc = correct / total
        print(f"Epoch {epoch+1} train acc: {acc:.4f}")
        torch.save(clf.state_dict(), save_path)

    return clf, labels_map


if __name__ == '__main__':
    # convenience CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['ae', 'clf'], default='ae')
    args = parser.parse_args()
    if args.mode == 'ae':
        train_ae()
    else:
        train_classifier()
