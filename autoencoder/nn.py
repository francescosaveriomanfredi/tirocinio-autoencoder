from typing import List, Optional, Callable
import torch
import pytorch_lightning as pl
from distributions import gaussian_sample
from torch import nn
import collections 

class MyFCLayers(torch.nn.Module):
    """
    A helper classto build fully-connected layers for a neural network.
    
    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    layers_dim
        The sequence of layers dimension
    
    """
    
    def __init__(
        self,
        n_in: int,
        n_out: int,
        layers_dim: List[int] = [128]
    ):
        super().__init__()
        layers_dim = [n_in] + layers_dim + [n_out]
        
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            # nn.Linear = layers.Dense
                            nn.Linear(
                                n_in,
                                n_out,
                            ),
                            nn.ReLU()
                        )
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )
    
    def forward(self, x:torch.Tensor):
        """
        Forward computation on ``x``

        Parameters
        ----------
        x
            tensor of value with shape ``(n_in,)``
        
        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        return self.fc_layers.forward(x)



class MyEncoder(nn.Module):
    """
    Encoder data of ``n_input`` dimensions into a laent space of ``n_output`` dimension

    Uses a fully-connected neural network with ``layers_dim`` units.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    layers_dim
        The list containing the number of layer's units
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance1
        Defaults to :meth:`torch.exp`.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        layers_dim: List[int] = [128],
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        ):
            super().__init__()

            self.var_eps = var_eps
            self.encoder = MyFCLayers(
                n_in=n_input,
                n_out=layers_dim[-1],
                layers_dim=layers_dim[:-1],
            )
            self.mean_encoder = nn.Linear(layers_dim[-1], n_output)
            self.var_encoder = nn.Linear(layers_dim[-1], n_output)
            self.var_activation = torch.exp if var_activation is None else var_activation
        
    def forward(self, x:torch.Tensor):
        r"""
        The forward computation for a single sample.
        #. Encodes the data into latent space using the encoder network
        #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
        #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        q_s = gaussian_sample(q_m, q_v.sqrt())

        return q_m, q_v, q_s

class MyDecoder(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimension.
    
    Uses a fully-connected neural network of ``layers_dim`` units.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    layers_dim
        The list containing the number of layer's units
    eps
        Minimum value for the mean and shape;
        used for numerical stability
    mean_activation
        Callable used to ensure positivity of the mean of NB distribution
        Defaults to :meth:`torch.exp`.
    shape_activation
        Callable used to ensure positivity of the shape of NB distribution
        Defaults to :meth:`torch.exp`.
    """
    def __init__(
        self,
        n_input: int,
        n_output: int,
        layers_dim: List[int] = [128],
        eps: float = 1e-4,
        mean_activation: Optional[Callable] = None,
        shape_activation: Optional[Callable] = None,
        ):
            super().__init__()
            
            self.eps=eps
            self.decoder = MyFCLayers(
                n_in=n_input,
                n_out=layers_dim[-1],
                layers_dim=layers_dim[:-1],
            )
            self.mean_decoder = nn.Linear(layers_dim[-1], n_output)
            self.shape_decoder = nn.Linear(layers_dim[-1], n_output)
            self.mean_activation = torch.exp if mean_activation is None else mean_activation
            self.shape_activation = torch.exp if mean_activation is None else shape_activation
        
    def forward(self, x:torch.Tensor, library:torch.Tensor):
        r"""
        The forward computation for a single sample.
        #. Decodes the data from latent space using the decoder network
        #. Generates a mean \\( p_m \\) and shape \\( p_r \\) of a multivariate NB distribution
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        library
            tensor with shape (1,)
        
        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_output,)`` for mean and shape
        """
        p = self.decoder(x)
        p_m = self.mean_activation(self.mean_decoder(p)) + self.eps
        p_r = self.shape_activation(self.shape_decoder(p)) + self.eps
        p_m = p_m * library
        return p_m, p_r



