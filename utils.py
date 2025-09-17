import torch
import torch.nn as nn
import torch.nn.functional as F

class EntropyLoss(nn.Module):
    """
    Entropy Loss for Uncertainty Maximization.
    This loss is used to make the classifier's predictions on out-of-domain
    data
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The output logits from the classifier.
        """
        # Apply softmax to get probabilities
        p = F.softmax(x, dim=1)
        # Log-softmax for numerical stability
        log_p = F.log_softmax(x, dim=1)
        # Entropy H(p) = -sum(p * log(p))
       
        return -torch.mean(torch.sum(p * log_p, dim=1))

def reparameterize(mu, log_var):
    """
    Reparameterization trick to sample from N(mu, var) from N(0,1).
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

def kl_divergence_loss(mu, log_var):
    """
    Calculates the KL divergence between a learned Gaussian distribution
    and a standard normal distribution N(0, I).
    This corresponds to L^KL in the paper for GLA.
    """
    # Formula for KL divergence between N(mu, var) and N(0, I)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kld.mean()
