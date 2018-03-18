import torch
import torch.nn as nn
from torch.autograd import Variable
import math

class Latentloss(nn.Module):
    """KL divergence for shared and private variables"""
    def __init__(self, n_h1, n_h2):
        super().__init__()
        self.n_h1 = n_h1
        self.n_h2 = n_h2


    def forward(self, z_mean, z_log_sigma_sq, prior_z_mean=None, prior_z_log_sigma_sq=None,h1_mean=None, h2_mean=None,
                h1_log_sigma_sq=None, h2_log_sigma_sq=None, prior_h1_mean=None, prior_h2_mean=None, prior_h1_log_sigma_sq=None,
                prior_h2_log_sigma_sq=None):
        subz = z_mean - prior_z_mean
        logz = torch.log(prior_z_log_sigma_sq)*2
        expz = torch.exp(logz)
        latent_loss_z = -0.5 * (1 + z_log_sigma_sq - logz - subz**2/expz
                       - torch.exp(z_log_sigma_sq)/expz).sum()

        if self.n_h1>0:
            subh1 = h1_mean - prior_h1_mean
            logh1 = torch.log(prior_h1_log_sigma_sq)*2
            exph1 = torch.exp(logh1)
            latent_loss_h1 = -0.5 * (1 + h1_log_sigma_sq - logh1- subh1**2/exph1
                       - torch.exp(h1_log_sigma_sq)/exph1).sum()

        if self.n_h2>0:
            subh2 = h2_mean - prior_h2_mean
            logh2 = torch.log(prior_h2_log_sigma_sq)*2
            exph2 = torch.exp(logh2)
            latent_loss_h2 = -0.5 * (1 + h2_log_sigma_sq - logh2- subh2**2/exph2
                       - torch.exp(h2_log_sigma_sq)/exph2).sum()
        if self.n_h1>0:
            latent_loss = (latent_loss_z + latent_loss_h1 + latent_loss_h2).sum()
        else:
            latent_loss = latent_loss_z

        return latent_loss


class ReconstructionLoss(nn.Module):
    """Reconstruction Loss"""
    def __init__(self, losstype, STDVAR):
        super().__init__()
        self.losstype = losstype
        self.STDVAR = STDVAR

    def forward(self, x_input, x_reconstr_mean, x_reconstr_log_sigma_sq, n_out):

        if self.losstype==0:
            # Cross entropy loss
            reconstr_loss =- (x_input*torch.log(1e-6+x_reconstr_mean)+(1-x_input)*(torch.log(1e-6+1-x_reconstr_mean))).sum()
        elif self.losstype==1:
            # Least squares loss, with learned std.
            reconstr_loss = 0.5 * (
                torch.div((x_input - x_reconstr_mean)**2, 1e-6 + torch.exp(x_reconstr_log_sigma_sq))
                ).sum + 0.5 * x_reconstr_log_sigma_sq.sum() + 0.5 * math.log(2 * math.pi) * n_out
        elif self.losstype == 2:
            # Least squares loss, with specified std.
            reconstr_loss = 0.5 * (((x_input - x_reconstr_mean) / self.STDVAR)**2).sum() + 0.5 * math.log(
                2 * math.pi * self.STDVAR * self.STDVAR) * n_out

            # Average over the minibatch.
        cost = reconstr_loss.sum()
        return cost


