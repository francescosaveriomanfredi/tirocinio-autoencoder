import anndata as an
import numpy as np
from typing import Optional, Callable, Union, Optional, Sequence
from torch.utils import data
from scipy.sparse import issparse
from scipy.stats import binom
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

def filter_gene(
    adata:an.AnnData,
    gene_filter:Optional[str]="highly_variable",
    n_gene:Optional[int]=None,
):
    """
    Filter the AnnData based on its var field.

    parameters
    -----------
    adata 
        AnnData
    gene_filter
        Var field name used to filter the data.
    n_gene
        If None gene_filter var must contains boolean value
        else gene_filter var must contains integer value.
        high value is better. 
    """
    if gene_filter is None:
        gene_index = adata.var_names
    elif n_gene is None:
        gene_index = adata.var_names[adata.var[gene_filter]]
    else:
        gene_index = adata.var[gene_filter].nlargest(n_gene).index
    
    return adata[:, gene_index].copy()

class Scaler:
    def __init__(self, n, scaler):
        self.scaler = scaler.fit(n)
    def __call__(self, n):
        return self.scaler.transform(n)

class BinomDropout:
    def __init__(self, p):
        self.p = p
    def __call__(self, n):
        return n - binom.rvs(n, self.p)

class CountDataset(data.Dataset):
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
        adata: an.AnnData,
        gene_filter: Optional[str]="highly_variable",
        n_gene: Optional[int]=None,
        scale_type: Union[str, Callable, None]=None,
        p_gene_dropout: float=0.
    ):
        super().__init__()
        adata = filter_gene(adata, gene_filter, n_gene)
        self.data = adata.X
        self.dim = self.data.shape[0]
        if scale_type is None:
            self.scaler = lambda x:x
        elif scale_type == "StandardScaler":
            self.scaler = Scaler(self.data, StandardScaler(with_mean=False))
        elif scale_type == "MaxAbsScaler":
            self.scaler = Scaler(self.data, MaxAbsScaler())
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


class CountDataModule(pl.LightningDataModule):
    def __init__(
        self,
        adata:Union[an.AnnData, str],
        batch_size:int=128,
        n_gene:Optional[int]=None,
        gene_filter:Optional[str]="highly_variable",
        scale_type:Union[Callable,str,None]="StandardScaler",
        p_gene_dropout:float=0.,
        train_val_test_lengths:Sequence[int|float]=[0.7, 0.2, 0.1],
        num_workers:int=0,
    ):
        super().__init__()
        self.adata=adata
        self.batch_size = batch_size
        self.n_gene = n_gene
        self.gene_filter = gene_filter
        self.scale_type = scale_type
        self.p_gene_dropout = p_gene_dropout
        self.train_val_test_lengths = train_val_test_lengths
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        # If we give a path and not an AnnData
        if isinstance(self.adata, str):
            self.adata = an.read_h5ad(f"{self.adata}")
        
        self.count_predict = CountDataset(
            adata=self.adata,
            gene_filter=self.gene_filter,
            n_gene=self.n_gene,
            scale_type=self.scale_type,
            p_gene_dropout=self.p_gene_dropout
        )

        self.count_train, self.count_val, self.count_test = random_split(
            self.count_predict, 
            self.train_val_test_lengths
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.count_train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.count_val, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.count_test, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.count_predict, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )