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
        self.decoder = MyDecoder(n_latent,n_input, layers_dim)
        self.l_encoder = MyFCLayers(n_input, 1, library_layers_dim)

    def forward(self, x):
        return self.encoder(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        qz_m, qz_v, z = self.encoder(x) 
        l = self.l_encoder(x)+ 1e-3
        px_m, px_r = self.decoder(z, l)
        nb = -log_nb_positive(y, px_m, px_r)
        nb_loss = torch.mean(nb.sum(dim=-1))
        nb_metric = torch.mean(nb.mean(dim=-1))
        kl = kullback_normal_divergence(qz_m, qz_v.sqrt())
        kl_loss = torch.mean(kl.sum(dim=-1))
        kl_metric = torch.mean(kl.mean(dim=-1))
        loss = nb_loss + kl_loss
        self.log_dict({'kl_metric': kl_metric, "nb_metric": nb_metric}, prog_bar=True, on_epoch=True,on_step=False, batch_size=x.shape[0])
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        return self.forward(x) 