from typing import List
import torch
import pytorch_lightning as pl
from nn import MyFCLayers, MyEncoder, MyDecoder
from distributions import (
    kullback_normal_divergence,
    log_nb_positive
)

torch.backends.cudnn.benchmark = True

class AutoencoderNB(pl.LightningModule):
    """
    Simple variational autoencoder model.
    
    Parameters
    ----------
    n_input
      Number of input genes
    n_latent
      Dimensionality of the latent space
    layer_dim
      Number of units for each hidden layer both for encoder and decoder
    library_layers_dim
      Number of units for each hidden layer both for library encoder
    """
    def __init__(self, n_input, n_latent: int, layers_dim:List[int], library_layers_dim=List[int]):
        super().__init__()
        self.encoder = MyEncoder(n_input, n_latent, layers_dim)
        layers_dim.reverse()
        self.decoder = MyDecoder(n_latent,n_input, layers_dim)
        self.l_encoder = MyFCLayers(n_input, 1, library_layers_dim)

    def forward(self, x_scaled, training=False):
        qz_m, qz_v, z = self.encoder(x_scaled)
        l = self.l_encoder(x_scaled)+ 1e-4
        if training:
            px_m, px_r = self.decoder(z, l)
        else:
            px_m, px_r = self.decoder(qz_m, l)
        return qz_m, qz_v, z, px_m, px_r, l
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def nb_loss(self, x, px_m, px_r, log=None):
        gene_size=x.shape[1]
        nb_loss = torch.mean(-log_nb_positive(x, px_m, px_r).sum(dim=-1))
        if log is not None:
            self.log(f"{log}_nb_loss", nb_loss/gene_size)
        return nb_loss

    def kl_loss(self, qz_m, qz_v, log=None):
        gene_size=qz_m.shape[1]
        kl_loss = torch.mean(kullback_normal_divergence(qz_m, qz_v.sqrt()).sum(dim=-1))
        if log is not None:
            self.log(f"{log}_kl_loss", kl_loss/gene_size)
        return kl_loss

    def recall_metric(self, x, px_m, px_r, log=None):
        gene_size=x.shape[1]
        zeros = torch.zeros_like(x)
        zero_pred_proba = torch.exp(log_nb_positive(zeros, px_m, px_r))
        pred_zero = torch.where(zero_pred_proba > 0.5, 0, 1)
        true_zero = torch.where(x > 0, 1, 0)
        correct = (pred_zero * true_zero).sum(axis=-1)
        total = true_zero.sum(axis=-1)
        recall = (correct / total).mean()
        if log is not None:
            self.log(f"{log}_recall", recall)
        return recall
    
    def crossentropy_metric(self, x, px_m, px_r, log=None):
        gene_size=x.shape[1]
        zeros = torch.zeros_like(x)
        zero_pred_proba = torch.exp(log_nb_positive(zeros, px_m, px_r))
        true_proba = torch.where(x > 0, 1-zero_pred_proba, zero_pred_proba)
        true_proba = -torch.log(true_proba).mean()
        if log is not None:
            self.log(f"{log}_true_proba", true_proba)
        return true_proba
    
    def training_step(self, train_batch, batch_idx):
        x_scaled, x = train_batch
        qz_m, qz_v, z, px_m, px_r, l = self.forward(x_scaled, True)
        nb_loss = self.nb_loss(x, px_m, px_r, log="train")
        kl_loss = self.kl_loss(qz_m, qz_v, log="train")
        _ = self.recall_metric(x, px_m, px_r, log="train")
        _ = self.true_proba_metric(x, px_m, px_r, log="train")

        loss = nb_loss + kl_loss
        return loss

    def validation_step(self, val_batch, batch_idx):
        x_scaled, x = val_batch
        qz_m, qz_v, z, px_m, px_r, l = self.forward(x_scaled, False)
        nb_loss = self.nb_loss(x, px_m, px_r, log="val")
        kl_loss = self.kl_loss(qz_m, qz_v, log="val")
        _ = self.recall_metric(x, px_m, px_r, log="val")
        _ = self.true_proba_metric(x, px_m, px_r, log="val")
    
    def predict_step(self, batch, batch_idx):
        x_scaled, x = batch
        return self.forward(x_scaled) 