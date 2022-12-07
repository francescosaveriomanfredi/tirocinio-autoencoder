import anndata as an
import pytorch_lightning as pl
from typing import Optional
from torch.utils import data
from scipy.sparse import issparse

class AnnDataset(data.Dataset):
    def __init__(
        self,
        adata: an.AnnData,
        layer: Optional[str]=None,
        label: Optional[str]=None
    ):
        super().__init__()
        self.dim = adata.n_obs
        self.label = adata.obs_names if label is None else adata.obs[label]
        self.X = adata.X if layer is None else adata.layers[layer]
    
    def __len__(self):
        return self.dim
    
    def __getitem__(self, idx):
        x = self.X[idx].toarray() if issparse(self.X) else self.X[idx]
        x = x.squeeze(0)
        l = self.label[idx]
        return x, x
