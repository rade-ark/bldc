import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .features import load_audio, compute_mel
from .config import RAW_DIR, SR

class BLDCMelDataset(Dataset):
    """Assumes folder structure: data/raw/<label>/*.wav"""
    def __init__(self, file_list, labels_map, transform=None):
        self.file_list = file_list
        self.labels_map = labels_map
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        y = load_audio(path)
        mel = compute_mel(y)
        # shape: (n_mels, t) -> add channel
        mel = np.expand_dims(mel, axis=0)
        label_name = Path(path).parent.name
        label = self.labels_map[label_name]
        sample = {
            'mel': torch.tensor(mel),
            'label': torch.tensor(label, dtype=torch.long),
            'path': path
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def make_dataloaders(data_dir=RAW_DIR, batch_size=32, val_split=0.2, seed=42):
    # scan
    classes = [d.name for d in Path(data_dir).iterdir() if d.is_dir()]
    classes.sort()
    labels_map = {c: i for i, c in enumerate(classes)}

    all_files = []
    for c in classes:
        files = list((Path(data_dir) / c).glob('*.wav'))
        all_files.extend([str(p) for p in files])

    train_files, val_files = train_test_split(all_files, test_size=val_split, random_state=seed, stratify=[Path(f).parent.name for f in all_files])

    train_ds = BLDCMelDataset(train_files, labels_map)
    val_ds = BLDCMelDataset(val_files, labels_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, labels_map
