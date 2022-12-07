import torch

def gaussian_sample(
        mean: torch.Tensor,
        std: torch.Tensor
):
    """
    Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
    Parameters
    ----------
    mean
        A tensor of the mean of multidimensional normal distribution
    std
        A tensor of the var of multidimensional normal distribution
    
    Returns
    -------
        A tensor of the sampled variable of the same dimension of the given parameter
    
    """
    s = mean.shape
    epsilon = torch.normal(0, 1, size=s)
    epsilon=epsilon.to(mean.device)
    return mean + std * epsilon

def kullback_normal_divergence(z_mean, z_std):
    """
    Kullback divergence from the given gaussian distribution
    and the normal distribution.
    this operatio is derivable and can be use as kind of loss.
    """
    return -0.5 * (1 + torch.log(z_std)*2 - torch.square(z_mean) - z_std.square())

def log_nb_positive(x, mu, r, eps = 1e-8, log_fn = torch.log, lgamma_fn = torch.lgamma,):
    """
    Log likelihood (scalar) of a minibatch according to a nb model.
    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    r
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(r + mu + eps)
    res = (
        r * (log(r + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + r + eps)
        - lgamma(r + eps) 
        - lgamma(x + 1 + eps) 
    )

    return res