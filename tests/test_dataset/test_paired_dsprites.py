from vsa_encoder.dataset.paired_dsprites import Dsprites


class TestPairedDsprites:
    def test_dsprites_dataset(self):
        n_train, n_test = 100_000, 30_000
        md = Dsprites(max_exchanges=1,
                      path="/home/yessense/PycharmProjects/vsa-encoder/one_exchange/dsprite_train.npz")
        pairs = md.make_indices(n_train, n_test)
        assert len(pairs) == 4
        assert pairs[0].shape  == (n_train, 2)
        assert pairs[1].shape  == (n_train, 5)
        assert pairs[2].shape  == (n_test, 2)
        assert pairs[3].shape  == (n_test, 5)

