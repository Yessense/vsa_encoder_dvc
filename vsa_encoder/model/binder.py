from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Binder(nn.Module):
    n_features: int
    latent_dim: int
    hd_placeholders: nn.Parameter

    def __init__(self, n_features: int, latent_dim: int):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim


class RandnBinder(Binder):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__(n_features, latent_dim)
        hd_placeholders = torch.randn(1, self.n_features, self.latent_dim)
        self.hd_placeholders = nn.Parameter(data=hd_placeholders)

    def forward(self, z):
        return z * self.hd_placeholders.data


class FourierBinder(Binder):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__(n_features, latent_dim)
        hd_placeholders = torch.randn(1, self.n_features, self.latent_dim)
        norm = torch.linalg.norm(hd_placeholders, dim=-1)
        norm = norm.unsqueeze(-1).expand(hd_placeholders.size())
        hd_placeholders = hd_placeholders / norm
        self.hd_placeholders = nn.Parameter(data=hd_placeholders)

    def forward(self, z):
        out = torch.fft.irfft(torch.fft.rfft(self.hd_placeholders.data) * torch.fft.rfft(z), dim=-1)
        return out


class IdentityBinder(Binder):
    def __init__(self, n_features: int, latent_dim: int):
        super().__init__(n_features, latent_dim)

    def forward(self, z):
        return z


if __name__ == '__main__':
    n_features = 5
    latent_dim = 1024
    batch_size = 2

    fourier_binder = FourierBinder(n_features=n_features, latent_dim=latent_dim)
    z = torch.randn(batch_size, n_features, latent_dim)

    out = fourier_binder(z)

    print()
