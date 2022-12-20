import torch
from vsa_encoder.model.vsa_vae import VSAVAE


class TestVsaVae:
    def test_general_paired_dsprites(self):
        batch_size = 10

        vsavae = VSAVAE()

        x = torch.randn(batch_size, 1, 64, 64)
        exchanges = torch.randint(0, 2, (10, 5), dtype=bool).unsqueeze(-1)

        out = vsavae.forward(x, x, exchanges)
        # 3 objects
        assert len(out) == 3

        # 2 reconstructions
        assert len(out[0]) == 2
        assert out[0][0].shape == out[0][1].shape == (batch_size, 1, 64, 64)

    def test_general_clevr(self):
        batch_size = 10
        n_features = 6
        latent_dim = 1024

        vsavae = VSAVAE(n_features=n_features, image_size=(3, 128, 128), latent_dim=latent_dim)

        x = torch.randn(batch_size, 3, 128, 128)
        exchanges = torch.randint(0, 2, (batch_size, n_features), dtype=bool).unsqueeze(-1)

        out = vsavae.forward(x, x, exchanges)
        # 3 objects
        assert len(out) == 3

        # 2 reconstructions
        assert len(out[0]) == 2
        assert out[0][0].shape == out[0][1].shape == (batch_size, 3, 128, 128)
