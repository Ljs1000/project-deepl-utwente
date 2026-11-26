import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),   # 75x75
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),  # 37x37
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(), # 18x18
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU() # 9x9
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(256 * 9 * 9, latent_dim)
        self.fc_lv = nn.Linear(256 * 9 * 9, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 256 * 9 * 9)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(), # 18x18
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),  # 36x36
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),   # 72x72
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),   # 144x144
            nn.ConvTranspose2d(16, 1, 4, 1, 1), nn.Sigmoid()  # ~147x147 (slight mismatch)
        )

    def encode(self, x):
        h = self.flatten(self.enc(x))
        mu = self.fc_mu(h)
        lv = self.fc_lv(h)
        return mu, lv

    def reparameterize(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 256, 9, 9)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparameterize(mu, lv)
        x_hat = self.decode(z)
        return x_hat, mu, lv

def vae_loss(x, x_hat, mu, lv):
    # Resize mismatch guard (keeps skeleton simple)
    if x_hat.shape[-2:] != x.shape[-2:]:
        x_hat = F.interpolate(x_hat, size=x.shape[-2:], mode="bilinear", align_corners=False)
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon + 1e-3 * kld, recon, kld
