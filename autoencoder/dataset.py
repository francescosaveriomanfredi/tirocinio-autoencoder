import urllib.request 
import os
import anndata as an
import pandas as pd
import numpy as np
from typing import Optional, Callable, Union, Optional, Sequence
from torch.utils import data
from scipy.sparse import issparse
from scipy.stats import binom
import scanpy as sc
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import urllib.request

def download(filename, url, destination_dir, expected_bytes = None, force = False):
    if url is None:
      if filename == "count1":
            url = "https://drive.google.com/uc?export=download&id=1Mda9KAWnX4JBTm8EHTIfQ6DqxkT3lXsL"
    
    filepath = os.path.join(destination_dir, filename)

    if force or not os.path.exists(filepath):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        print('Attempting to download: ' + filename)
        filepath, _ = urllib.request.urlretrieve(url, filepath,)
        print('Download complete!')

    statinfo = os.stat(filepath)

    if expected_bytes != None:
        if statinfo.st_size == expected_bytes:
            print('Found and verified: ' + filename)
        else:
            raise Exception('Failed to verify: ' + filename + '. Can you get to it with a browser?')
    else:
        print('Found: ' + filename)
        print('The size of the file: ' + str(statinfo.st_size))

    return filepath

def filter_gene(
    adata:an.AnnData,
    gene_filter:Optional[str]="highly_variable",
    n_gene:Optional[int]=None,
    gene_filter_ascending=True,
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
    gene_filter_ascending
         If select largest or smallest based on value
    """
    if gene_filter_ascending:
        n_row = pd.Series.nlargest
    else:
        n_row = pd.Series.nsmallest
    if gene_filter is None:
        gene_index = adata.var_names
    elif n_gene is None:
        gene_index = adata.var_names[adata.var[gene_filter]]
    else:
        gene_index = n_row(adata.var[gene_filter], n_gene).index
    
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

class Binarize:
    def __init__(self):
        pass
    def __call__(self, n):
        return np.where(n > 0, 1, 0)

class CountDataset(data.Dataset):
    """
    A Dataset of genes count
    
    Parameters
    ----------
    data_path
        path to the count gene expression adata
    scale_type
        A callable used to scale data or
        StandardScaler, BinarizeScaller
    p_gene_dropout
        probability to drop one gene espression for each gene expression
    n_gene
        the number of gene select using gene_filter var field
    gene_filter
         a adata var field using to select the gene
    gene_filter_ascending
         the order to select the genes based on gene_filter
    
    """
    def __init__(
        self,
        adata_path: str,
        scale_type: Union[str, Callable, None]=None,
        p_gene_dropout: float=0. ,
        n_gene:Optional[int]=None,
        gene_filter:Optional[str]=None,
        gene_filter_ascending:bool=True,
        #url:Optional[str]=None
    ):
        super().__init__()
        self.adata = an.read_h5ad(adata_path)
        self.adata = filter_gene(
            self.adata, 
            gene_filter, 
            n_gene,
            gene_filter_ascending
        )
        self.dim = self.adata.n_obs
        if scale_type is None:
            self.scaler = lambda x:x
        elif scale_type == "StandardScaler":
            self.scaler = Scaler(self.adata.X, StandardScaler(with_mean=False))
        elif scale_type == "BinarizeScaler":
            self.scaler = Binarize()
        else:
             self.scaler = scale_type
        self.gene_dropout = BinomDropout(p_gene_dropout)
        
    def __len__(self):
        return self.dim
    
    def __getitem__(self, idx):
        out_data = self.adata.X[idx].toarray() if issparse(self.adata.X) else self.data.X[idx]
        out_data = self.gene_dropout(out_data.astype(np.int32)).astype(np.float32)
        in_data = self.scaler(out_data).astype(np.float32)
        return {
            "X":in_data.squeeze(0), 
            "Y":out_data.squeeze(0)
        }


class CountDataModule(pl.LightningDataModule):
    """
    Parameters
    ----------
    data_path
        path to the count gene expression adata
    batch_size
        the batch size of data to feed the module
    n_gene
        the number of gene select using gene_filter var field
    gene_filter
         a adata var field using to select the gene
    gene_filter_ascending
         the order to select the genes based on gene_filter
    scale_type
        A callable used to scale data or
        StandardScaler, BinarizeScaller
    p_gene_dropout
        probability to drop one gene espression for each gene expression
    train_val_test_lengths
        The train validation test partition to split data
    num_workers
        The number of thread to use.
    """
    def __init__(
        self,
        adata_path:str,
        batch_size:int=128,
        n_gene:Optional[int]=None,
        gene_filter:Optional[str]=None,
        gene_filter_ascending:bool=True,
        scale_type:Union[Callable,str,None]="StandardScaler",
        p_gene_dropout:float=0.,
        train_val_test_lengths:Sequence[float]=[0.7, 0.2, 0.1],
        num_workers:int=0,
        #source:Optional[str]=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.adata_path=adata_path
        self.batch_size = batch_size
        self.n_gene = n_gene
        self.gene_filter = gene_filter
        self.gene_filter_ascending = gene_filter_ascending
        self.scale_type = scale_type
        self.p_gene_dropout = p_gene_dropout
        self.train_val_test_lengths = train_val_test_lengths
        self.num_workers = num_workers
        #self.source=source
    
    def prepare_data(self):
        # download
        pass
    
    def setup(self, stage=None):
        self.count_predict = CountDataset(
            adata_path=self.adata_path,
            scale_type=self.scale_type,
            p_gene_dropout=self.p_gene_dropout,
            gene_filter=self.gene_filter, 
            n_gene=self.n_gene,
            gene_filter_ascending=self.gene_filter_ascending
        )
        self.adata = self.count_predict.adata

        self.count_train, self.count_val, self.count_test = random_split(
            self.count_predict, 
            self.train_val_test_lengths
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.count_train, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
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

def highly_variable_genes(adata, n_top_genes):
    adata_ = adata.copy()
    sc.pp.log1p(adata_) # log(expression + 1)
    sc.pp.highly_variable_genes(
        adata_,
        n_top_genes=n_top_genes,
        flavor="seurat"
    )
    adata_.X = adata.X
    return adata_

def unbatch(batched_param):
    """
    utility function used to 
    unbatch the result of a prediction.
    """
    n_param = len(batched_param[0])
    unbatched_param = []
    for param in range(n_param):
        unbatched_param.append(
            np.vstack([batch[param] for batch in batched_param])
        )
    return unbatched_param

def annotate_predictions(adata, layers, obsm, obs,):
    """
    Add information to the adata passed
    """
    for layers_label in layers:
        adata.layers[layers_label] = layers[layers_label]
    for obsm_label in obsm:
        adata.obsm[obsm_label] = obsm[obsm_label]
    for obs_label in obs:
        adata.obs[obs_label] = obs[obs_label]

