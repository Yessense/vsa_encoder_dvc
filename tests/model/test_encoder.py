from vsa_encoder.model.encoder import Encoder
import torch



class TestEncoder:
    def test_paired_dsprites(self):
        # Paired_dsprites
        image_size = (1, 64, 64)
        n_features = 5
        latent_dim = 1024
        batch_size = 2
        encoder = Encoder(image_size=image_size, n_features=n_features, latent_dim=latent_dim)

        x = torch.randn(batch_size, 1, 64, 64)

        out = encoder(x)

        assert len(out) == 2
        # mus
        assert out[0].shape == (batch_size, latent_dim * n_features)
        # log_sigma
        assert out[1].shape == (batch_size, latent_dim * n_features)

    def test_clevr(self):
        # Clevr
        image_size = (3, 128, 128)
        n_features = 6
        latent_dim = 1024
        batch_size = 2
        encoder = Encoder(image_size=image_size, n_features=n_features, latent_dim=latent_dim)

        x = torch.randn(batch_size, 3, 128, 128)

        out = encoder(x)

        assert len(out) == 2
        # mus
        assert out[0].shape == (batch_size, latent_dim * n_features)
        # log_sigma
        assert out[1].shape == (batch_size, latent_dim * n_features)


