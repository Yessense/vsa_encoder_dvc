import torch

from vsa_encoder.model.binder import FourierBinder, RandnBinder, IdentityBinder

class TestBinder:
    def test_fourier_binder(self):
        n_features = 5
        latent_dim = 1024
        batch_size = 2

        fourier_binder = FourierBinder(n_features=n_features, latent_dim=latent_dim)
        z = torch.randn(batch_size, n_features, latent_dim)

        out = fourier_binder(z)
        assert out.shape == z.shape

    def test_randn_binder(self):
        n_features = 5
        latent_dim = 1024
        batch_size = 2

        randn_binder = RandnBinder(n_features=n_features, latent_dim=latent_dim)
        z = torch.randn(batch_size, n_features, latent_dim)

        out = randn_binder(z)

        assert out.shape == z.shape

    def test_identity_binder(self):
        n_features = 5
        latent_dim = 1024
        batch_size = 2

        randn_binder = IdentityBinder(n_features=n_features, latent_dim=latent_dim)
        z = torch.randn(batch_size, n_features, latent_dim)

        out = randn_binder(z)

        assert torch.all(out == z)




