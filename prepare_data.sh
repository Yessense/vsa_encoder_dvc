python3 -m vsa_encoder.dataset.paired_dsprites --mode make_dataset --max_exchanges 1\
  --train_size 100000 --test_size 30000\
  --path_to_dsprites_train "./data/paired_dsprites/dsprites_train.npz"\
  --save_path "./data/paired_dsprites/"
