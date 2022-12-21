import math
from dataclasses import dataclass
from typing import Tuple, Optional

import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.optim import lr_scheduler

from .binder import FourierBinder, RandnBinder, IdentityBinder, Binder
from .decoder import Decoder
from .encoder import Encoder
from ..utils import iou_pytorch
from .configs import VAEConfig


class VSAVAE(pl.LightningModule):
    binder: Binder
    cfg: VAEConfig

    def __init__(self,
                 cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg

        # Experiment options
        self.image_size = cfg.model.image_size
        self.latent_dim = cfg.model.latent_dim
        self.n_features = cfg.model.n_features
        self.bind_mode = cfg.model.bind_mode
        self.normalization = None  # sum normalization on number of features

        # model parameters
        self.lr = cfg.model.lr
        self.kld_coef = cfg.model.kld_coef

        # Layers
        self.encoder = Encoder(latent_dim=self.latent_dim,
                               image_size=self.image_size,
                               n_features=self.n_features)
        self.decoder = Decoder(latent_dim=self.latent_dim,
                               image_size=self.image_size)

        # hd placeholders
        if self.bind_mode == 'fourier':
            self.binder = FourierBinder(self.n_features, self.latent_dim)
        elif self.bind_mode == 'randn':
            self.binder = RandnBinder(self.n_features, self.latent_dim)
        elif self.bind_mode == 'default':
            self.binder = IdentityBinder(self.n_features, self.latent_dim)
        else:
            raise ValueError(f"Wrong bind mode {self.bind_mode}")

        self.save_hyperparameters()

    def reparametrize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def get_features(self, x):
        """ Get features with shape -> [batch_size, n_features, latent_dim]"""
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)

        z = z.reshape(-1, self.n_features, self.latent_dim)
        return z

    def encode(self, x):
        """ Get features multiplied by placeholders

        x -> [batch_size, n_channels, height, width]
        out -> [batch_size, n_features, latent_dim]
        """
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)

        z = z.reshape(-1, self.n_features, self.latent_dim)

        z = self.binder(z)

        return z, mu, log_var

    def exchange(self, image_features, donor_features, exchange_labels):
        # Exchange
        exchange_labels = exchange_labels.expand(image_features.size())

        # Reconstruct image
        donor_features_exept_one = torch.where(exchange_labels, image_features,
                                               donor_features)
        donor_features_exept_one = torch.sum(donor_features_exept_one, dim=1)
        if self.normalization == 'n_features':
            donor_features_exept_one = donor_features_exept_one / math.sqrt(
                self.n_features)
        elif self.normalization == 'linalg_norm':
            norm = torch.linalg.norm(donor_features_exept_one,
                                     dim=-1).unsqueeze(-1)
            donor_features_exept_one = donor_features_exept_one / norm

        # Donor image
        image_features_exept_one = torch.where(exchange_labels, donor_features,
                                               image_features)
        image_features_exept_one = torch.sum(image_features_exept_one, dim=1)

        # Normalization
        if self.normalization == 'n_features':
            image_features_exept_one = image_features_exept_one / math.sqrt(
                self.n_features)
        elif self.normalization == 'linalg_norm':
            norm = torch.linalg.norm(image_features_exept_one,
                                     dim=-1).unsqueeze(-1)
            image_features_exept_one = image_features_exept_one / norm

        return donor_features_exept_one, image_features_exept_one

    def forward(self, image, donor, exchange_labels):
        image_features, image_mu, image_log_var = self.encode(image)
        donor_features, donor_mu, donor_log_var = self.encode(donor)

        donor_features_exept_one, image_features_exept_one = self.exchange(
            image_features, donor_features,
            exchange_labels)
        recon_like_image = self.decoder(donor_features_exept_one)
        recon_like_donor = self.decoder(image_features_exept_one)

        reconstructions = (recon_like_image, recon_like_donor)
        mus = (image_mu, donor_mu)
        log_vars = (image_log_var, donor_log_var)
        return reconstructions, mus, log_vars

    def _step(self, batch, batch_idx, mode='Train'):
        """ Base step"""

        # Logging period
        # Log Train samples once per epoch
        # Log Validation images triple per epoch
        if mode == 'Train':
            log_images = lambda x: x == 0
        elif mode == 'Validation':
            log_images = lambda x: x % 10 == 0
        elif mode == 'Test':
            log_images = lambda x: True
        else:
            raise ValueError

        image, donor, exchange_labels = batch
        reconstructions, mus, log_vars = self.forward(image, donor,
                                                      exchange_labels)

        mus = sum(mus) * 2 ** -0.5
        log_vars = sum(mus) * 2 ** -0.5

        image_loss, donor_loss, kld_loss = self.loss_f((image, donor),
                                                       reconstructions, mus,
                                                       log_vars)

        total_loss = (image_loss + donor_loss) * 0.5 + self.kld_coef * kld_loss

        iou_image = iou_pytorch(reconstructions[0], image)
        iou_donor = iou_pytorch(reconstructions[1], donor)
        total_iou = (iou_image + iou_donor) / 2

        # ----------------------------------------
        # Logs
        # ----------------------------------------

        self.log(f"{mode}/Total", total_loss)
        self.log(f"{mode}/Reconstruct Image", image_loss)
        self.log(f"{mode}/Reconstruct Donor", donor_loss)
        self.log(f"{mode}/Mean Reconstruction", (image_loss + donor_loss) / 2)
        self.log(f"{mode}/KLD", kld_loss * self.kld_coef)
        self.log(f"{mode}/iou total", total_iou)
        self.log(f"{mode}/iou image", iou_image)
        self.log(f"{mode}/iou donor", iou_donor)

        if log_images(batch_idx):
            self.logger.experiment.log({
                f"{mode}/Images": [
                    wandb.Image(image[0], caption='Image'),
                    wandb.Image(donor[0], caption='Donor'),
                    wandb.Image(reconstructions[0][0],
                                caption='Recon like Image'),
                    wandb.Image(reconstructions[1][0],
                                caption='Recon like Donor'),
                ]})

        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, mode='Train')
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, mode='Validation')

    def loss_f(self, gt_images, reconstructions, mus, log_vars):
        reduction = 'sum'
        loss = nn.MSELoss(reduction=reduction)
        image_loss = loss(reconstructions[0], gt_images[0])
        donor_loss = loss(reconstructions[1], gt_images[1])

        kld_loss = -0.5 * torch.sum(1 + log_vars - mus.pow(2) - log_vars.exp())

        if reduction == 'mean':
            batch_size = mus.shape[0]
            kld_loss = kld_loss / batch_size

        return image_loss, donor_loss, kld_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                            epochs=self.cfg.experiment.max_epochs,
                                            steps_per_epoch=self.cfg.experiment.steps_per_epoch,
                                            pct_start=0.2)
        return {"optimizer": optimizer,
                "lr_scheduler": {'scheduler': scheduler,
                                 'interval': 'step',
                                 'frequency': 1}, }
