import anndata as an
from typing import Optional
from torch.utils import data
from scipy.sparse import issparse

class MyAnDataset(data.Dataset):
    """
    A Dataset built from a layer of an anndata.
    The y is the raw counts and x is the same counts 
    scaled or not depending on ``layer_scale``
    
    Parameters
    ----------
    adata
        AnnData object
    layer_count
        If provided, which element of layers are raw counts
    layer_scale
        If provided, which element of layers are scaled counts
        (if not provided it use the same data of layer_count)
    
    """
    def __init__(
        self,
        adata: an.AnnData,
        layer_count: Optional[str] = None,
        layer_scale: Optional[str] = None,
        # label: Optional[str]=None
    ):
        super().__init__()
        self.dim = adata.n_obs
        # self.label = adata.obs_names if label is None else adata.obs[label]
        self.x = adata.X if layer_scale is None else adata.layers[layer_scale]
        self.y = self.x if layer_count is None else adata.layers[layer_count]
        
    def __len__(self):
        return self.dim
    
    def __getitem__(self, idx):
        y = self.y[idx].toarray().squeeze(0) if issparse(self.y) else self.y[idx]
        # l = self.label[idx]
        # if it is not scale just return the same data
        x = y if self.x is self.y else self.x[idx] 
        return x, y
