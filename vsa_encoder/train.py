from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import pytorch_lightning as pl

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from .model.configs import VAEConfig
from .dataset.clevr import PairedCogentClevr
from .dataset.paired_dsprites import PairedDspritesDataset
from .model.vsa_vae import VSAVAE

cs = ConfigStore.instance()
cs.store(name='config', node=VAEConfig)


@hydra.main(version_base=None, config_path="../conf/", config_name="train_config")
def train(cfg: VAEConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.experiment.seed)

    if cfg.dataset.mode == 'dsprites':
        cfg.model.image_size = (1, 64, 64)
        images_path = Path(cfg.dataset.path_to_dataset) / 'dsprites_train.npz'
        train_path = Path(cfg.dataset.path_to_dataset) / 'paired_train.npz'
        test_path = Path(cfg.dataset.path_to_dataset) / 'paired_test.npz'

        train_dataset = PairedDspritesDataset(dsprites_path=images_path,
                                              paired_dsprites_path=train_path)
        test_dataset = PairedDspritesDataset(dsprites_path=images_path,
                                             paired_dsprites_path=test_path)

        train_loader = DataLoader(train_dataset, batch_size=cfg.experiment.batch_size,
                                  num_workers=10, drop_last=True,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.experiment.batch_size, num_workers=10,
                                 drop_last=True)

    elif cfg.dataset.mode == 'clevr':
        cfg.model.image_size = (3, 128, 128)
        train_path = Path(cfg.dataset.path_to_dataset) / 'train'
        val_path = Path(cfg.dataset.path_to_dataset) / 'val'

        train_dataset = PairedCogentClevr(dataset_path=train_path)
        test_dataset = PairedCogentClevr(dataset_path=val_path)

        train_loader = DataLoader(train_dataset, batch_size=cfg.experiment.batch_size,
                                  num_workers=10,
                                  drop_last=True,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.experiment.batch_size, num_workers=10,
                                 drop_last=True)
    else:
        raise ValueError("Wrong mode")

    cfg.experiment.steps_per_epoch = len(train_loader)

    autoencoder = VSAVAE(cfg=cfg)

    top_metric_callback = ModelCheckpoint(monitor=cfg.model.monitor,
                                          save_top_k=cfg.checkpoints.save_top_k)
    every_epoch_callback = ModelCheckpoint(every_n_epochs=cfg.checkpoints.every_k_epochs)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [
        top_metric_callback,
        every_epoch_callback,
        lr_monitor,
    ]
    devices = [cfg.experiment.devices]

    wandb_logger = WandbLogger(project=cfg.dataset.mode + '_vsa',
                               name=f'{cfg.dataset.mode} -l {cfg.model.latent_dim} '
                                    f'-s {cfg.experiment.seed} -kl {cfg.model.kld_coef} '
                                    f'-bs {cfg.experiment.batch_size} '
                                    f'vsa',
                               log_model=True)

    # trainer
    trainer = pl.Trainer(accelerator=cfg.experiment.accelerator,
                         devices=devices,
                         max_epochs=cfg.experiment.max_epochs,
                         profiler=cfg.experiment.profiler,
                         callbacks=callbacks,
                         logger=wandb_logger,
                         check_val_every_n_epoch=cfg.checkpoints.check_val_every_n_epochs,
                         gradient_clip_val=cfg.experiment.gradient_clip)

    trainer.fit(autoencoder,
                train_dataloaders=train_loader,
                val_dataloaders=test_loader,
                ckpt_path=cfg.checkpoints.ckpt_path)


if __name__ == '__main__':
    train()
