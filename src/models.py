import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()
        self._dummy_conv_out = None
        self.fc_enc = None
        self.fc_dec = None
        self.latent_dim = latent_dim

    def _init_fc(self, x):
        # run a single forward to get sizes
        with torch.no_grad():
            z = self.enc(x)
            self._dummy_conv_out = z.shape
            n = z.numel() // z.shape[0]
            self.fc_enc = nn.Linear(n, self.latent_dim)
            self.fc_dec = nn.Linear(self.latent_dim, n)
            # create decoder conv transpose stack
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            )

    def forward(self, x):
        # x: (B, C, F, T)
        enc = self.enc(x)
        if self.fc_enc is None:
            self._init_fc(x)
        b = enc.shape[0]
        flat = enc.view(b, -1)
        z = self.fc_enc(flat)
        # decode
        dec_flat = self.fc_dec(z)
        dec = dec_flat.view(enc.shape)
        out = self.dec(dec)
        return out, z


class ClassifierFromLatent(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, z):
        return self.classifier(z)
