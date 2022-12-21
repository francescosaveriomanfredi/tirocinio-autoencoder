import anndata as an
import numpy as np
from typing import Optional, Callable, Union
from torch.utils import data
from scipy.sparse import issparse
from scipy.stats import binom
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

class Scaler:
    def __init__(self, n, scaler):
        self.scaler = scaler.fit(n)
    def __call__(self, n):
        return self.scaler.transform(n)

class BinomDropout:
    def __init__(self, p):
        self.p = p
    def __call__(self, n):
        n
        return n - binom.rvs(n, self.p)

class MyDataset(data.Dataset):
    """
    A Dataset of genes count
    
    Parameters
    ----------
    data
        raw counts in a sparse or not sparse format
    scale
        A callable used to scale data 
    p_gene_dropout
        probability to drop one gene espression for each gene expression
    """
    def __init__(
        self,
        data: any,
        scale_type: Union[str, Callable] = "StandardScaler",
        p_gene_dropout: float = 0.
    ):
        super().__init__()
        self.data = data
        self.dim = data.shape[0]
        if scale_type is None:
            self.scaler = lambda x:x
        elif scale_type == "StandardScaler":
            self.scaler = Scaler(data, StandardScaler(with_mean=False))
        elif scale_type == "MaxAbsScaler":
            self.scaler = Scaler(data, MaxAbsScaler())
        else:
             self.scaler = scale_type
        self.gene_dropout = BinomDropout(p_gene_dropout)
        
    def __len__(self):
        return self.dim
    
    def __getitem__(self, idx):
        out_data = self.data[idx].toarray() if issparse(self.data) else self.data[idx]
        out_data = self.gene_dropout(out_data.astype(np.int32)).astype(np.float32)
        in_data = self.scaler(out_data).astype(np.float32)
        return in_data.squeeze(0), out_data.squeeze(0)