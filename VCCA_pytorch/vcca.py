import torch
import torch.nn as nn
from torch.autograd import Variable
from Network import Network
from Network_reconstruction import Network_reconstruction

class VCCA(nn.Module):
    """Container module with VCCA"""
    def __init__(self, architecture, losstype1, losstype2, keepprob=0.001,l2_penalty=0.0, latent_penalty=1.0, STDVAR1=1.0, STDVAR2=1.0):
        super(VCCA, self).__init__()
        self.architecture = architecture
        self.l2_penalty = l2_penalty
        #self.learning_rate = learning_rate
        self.L=num_samples=5
        self.n_input1 = architecture["n_input1"]
        self.n_input2 = architecture["n_input2"]
        self.n_z  = architecture["n_z"]
        self.n_h1  = architecture["n_h1"]
        self.n_h2 = architecture["n_h2"]
        self.n_HF =  architecture["n_HF"]
        self.n_HG1  = architecture["n_HG1"]
        self.n_HG2  = architecture["n_HG2"]
        # Trade-off parameter for KL divergence.
        self.latent_penalty = latent_penalty
        # Gaussian standard variation for the observation models of each view, only matters for losstype=2.
        self.STDVAR1 = STDVAR1
        self.STDVAR2 = STDVAR2
        self.bsize = 1
        self.dropout = nn.Dropout(keepprob)
        # View 1 recognition network F

        # F hidden
        self.F_hidden, input_s = self.make_layer(self.architecture, "F_hidden_widths", self.n_input1)
        # F Gaussian
        # mean
        self.F_mean, input_G_m = self.make_layer(self.architecture, "F_Gaussian", input_s)
        # sigma
        self.F_sigma, input_G_s = self.make_layer(self.architecture,"F_Gaussian", input_s)
        # Private network for view 1

        # G1 hidden
        if self.n_h1 > 0:
            self.G1_hidden, input_G1 = self.make_layer(self.architecture, "G1_hidden_widths", self.n_input1)

            # G1 Gaussian
            # mean
            self.G1_mean, input_G1_m = self.make_layer(self.architecture, "G1_Gaussian", input_G1)
            # sigma
            self.G1_sigma, input_G1_s = self.make_layer(self.architecture, "G1_Gaussian", input_G1)

        # Private network for view 2

        # G2 hidden
        if self.n_h2 > 0:
            self.G2_hidden, input_G2 = self.make_layer(self.architecture, "G2_hidden_widths", self.n_input2)
            # G2 Gaussian
            # mean
            self.G2_mean, input_G2_m = self.make_layer(self.architecture, "G2_hidden_widths", input_G2)
            # sigma
            self.G2_sigma, input_G2_m = self.make_layer(self.architecture, "G2_hidden_widths", input_G2)

        # View 1 reconstruction network H1

        # H1 hidden
        if self.n_h1 > 0:
            temp_width1 = self.n_z + self.n_h1
            self.H1_hidden, input_H1 = self.make_construction_layer(self.architecture, "H1_hidden_widths", temp_width1)
        else:
            self.H1_hidden, input_H1 = self.make_construction_layer(self.architecture, "H1_hidden_widths",self.n_z)

        # View 2 reconstruction network H2
        if self.n_h2 > 0:
            temp_width2 = self.n_z + self.n_h2
            self.H2_hidden, input_H2 = self.make_construction_layer(self.architecture, "H2_hidden_widths", temp_width2)
        else:
            self.H2_hidden, input_H2 = self.make_construction_layer(self.architecture, "H2_hidden_widths", self.n_z)

    def forward(self, x, y):
        out = self.F_hidden(x)
        z_mean = self.F_mean(out)
        z_log_sigma_sq = self.F_sigma(out)
        z1 = self.draw_sample(z_mean)
        z2 = self.draw_sample(z_mean)
        if self.n_h1>0:
            out1 = self.G1_hidden(x)
            h1_mean = self.G1_mean(out1)
            h1_log_sigma_sq = self.G1_sigma(out1)
            h1 = self.draw_sample(h1_mean)
            act = torch.cat((z1, h1), 0)
            x1_reconstr_mean_from_z1, x1_reconstr_log_sigma_from_z1 = self.H1_hidden(act)
        else:

            x1_reconstr_mean_from_z1, x1_reconstr_log_sigma_from_z1 = self.H1_hidden(z1)

        if self.n_h2>0:
            out2 = self.G2_hidden(y)
            h2_mean = self.G2_mean(out2)
            h2_log_sigma_sq = self.G2_sigma(out2)
            h2 = self.draw_sample(h2_mean)
            act2 = torch.cat((z2, h2), 0)
            x2_reconstr_mean_from_z2, x2_reconstr_log_sigma_from_z2 = self.H1_hidden(act2)
        else:
            x2_reconstr_mean_from_z2, x2_reconstr_log_sigma_from_z2 = self.H1_hidden(z2)

        if self.n_h1>0:
            return z_mean, z_log_sigma_sq, h1_mean, h1_log_sigma_sq, h2_mean, h2_log_sigma_sq,\
        x1_reconstr_mean_from_z1,x1_reconstr_log_sigma_from_z1,x2_reconstr_mean_from_z2,\
        x2_reconstr_log_sigma_from_z2
        else:
            return z_mean, z_log_sigma_sq, x1_reconstr_mean_from_z1,x1_reconstr_log_sigma_from_z1,\
        x2_reconstr_mean_from_z2,x2_reconstr_log_sigma_from_z2


    def make_layer(self, architecture,width_type, input_size):
        input_s = input_size
        layers = []
        for i in range(len(architecture[width_type])):
            hidden_size = architecture[width_type][i]
            layers.append(Network(input_s, hidden_size))
            input_s = architecture[width_type][i]
        return nn.Sequential(*layers), input_s

    def make_construction_layer(self, architecture, width_type, input_size):
        input_s = input_size
        layers = []
        n = len(architecture[width_type])-1
        for i in range(len(architecture[width_type])):
            hidden_size = architecture[width_type][i]
            if i == n:
                layers.append(Network_reconstruction(input_s, hidden_size))
            else:
                layers.append(Network(input_s, hidden_size))
            input_s = architecture[width_type][i]
        return nn.Sequential(*layers), input_s

    def draw_sample(self, t_mean):
        z_epsshape = torch.addcmul(t_mean.size(), [self.L, 1])
        eps = torch.normal(0, 1, z_epsshape)
        t = torch.add(t_mean.repeat(self.L), torch.addcmul(torch.exp(0.5 * self.z_log_sigma_sq).repeat(self.L), eps))
        return t
