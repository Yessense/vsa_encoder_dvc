import torch

from vsa_encoder.model.decoder import Decoder


class TestDecoder:
    def test_clevr(self):
        # Clevr
        image_size = (3, 128, 128)
        latent_dim = 1024
        batch_size = 2
        decoder = Decoder(image_size=image_size, latent_dim=latent_dim)

        x = torch.randn(batch_size, 1024)

        out = decoder(x)

        assert out.shape == (batch_size, 3, 128, 128)

    def test_paired_dsprites(self):
        # Dsprites
        image_size = (1, 64, 64)
        latent_dim = 1024
        batch_size = 2
        decoder = Decoder(image_size=image_size, latent_dim=latent_dim)

        x = torch.randn(batch_size, 1024)
        out = decoder(x)

        assert out.shape == (batch_size, 1, 64, 64)
