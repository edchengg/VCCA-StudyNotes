# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
from loss import *
import vcca
from myreadinput import read_xrmb

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--Z", default=70, help="Dimensionality of features", type=int)
parser.add_argument("--H1", default=10, help="Dimensionality of private variables for view 1", type=int)
parser.add_argument("--H2", default=10, help="Dimensionality of private variables for view 2", type=int)
parser.add_argument("--HF", default=64, help="Dimensionality of BiLSTM for F", type=int)
parser.add_argument("--HG1", default=64, help="Dimensionality of BiLSTM for G1", type=int)
parser.add_argument("--HG2", default=64, help="Dimensionality of BiLSTM for G2", type=int)
parser.add_argument("--IM", default=0.0, help="Regularization constant for the IM penalty", type=float)
parser.add_argument("--stdvar1", default=1.0, help="Standard variation of view 1 observation", type=float)
parser.add_argument("--stdvar2", default=0.1, help="Standard variation of view 2 observation", type=float)
parser.add_argument("--dropprob", default=0.2, help="Dropout probability of networks.", type=float)
parser.add_argument("--zpenalty", default=1.0, help="Latent penalty for the KL divergence.", type=float)
parser.add_argument("--checkpoint", default="./vcca_xrmb", help="Path to saved models", type=str)
args=parser.parse_args()



# Set random seeds.
np.random.seed(0)
torch.manual_seed(0)

# Obtain parsed arguments.
Z = args.Z
print("Dimensionality of shared variables: %d" % Z)
H1 = args.H1
print("Dimensionality of view 1 private variables: %d" % H1)
H2 = args.H2
print("Dimensionality of view 2 private variables: %d" % H2)
HF = args.HF
HG1 = args.HG1
HG2 = args.HG2
IM_penalty = args.IM
print("Regularization constant for IM penalty: %f" % IM_penalty)
dropprob = args.dropprob
print("Dropout rate: %f" % dropprob)
stdvar1 = args.stdvar1
print("View 1 observation std: %f" % stdvar1)
stdvar2 = args.stdvar2
print("View 2 observation std: %f" % stdvar2)
latent_penalty = args.zpenalty
print("Latent penalty: %f" % latent_penalty)
checkpoint = args.checkpoint
print("Trained model will be saved at %s" % checkpoint)

# Some configurations.
losstype1 = 2  # Gaussian with given stdvar1.
losstype2 = 2  # Gaussian with given stdvar2.
learning_rate = 0.0001
l2_penalty = 0.0


# Define network architectures.
network_architecture = dict(
    n_input1=39 * 71,  # XRMB data MFCCs input
    n_input2=16 * 71,  # XRMB data articulation input
    n_z=Z,  # Dimensionality of shared latent space
    n_h1=H1,  # Dimensionality of individual latent space of view 1
    n_h2=H2,  # Dimensionality of individual latent space of view 2
    n_HF=HF,  # Dimensionality of LSTM-F
    n_HG1=HG1,  # Dimensionality of LSTM-G1
    n_HG2=HG2,  # Dimensionality of LSTM-G2
    F_hidden_widths=[1024, 1024],
    F_hidden_activations=[F.relu, F.relu, None],
    G1_hidden_widths=[1024, 1024],
    G1_hidden_activations=[F.relu, F.relu, None],
    G2_hidden_widths=[1024, 1024],
    G2_hidden_activations=[F.relu, F.relu, None],
    H1_hidden_widths=[1500, 1024, 1024, 39 * 71],
    H1_hidden_activations=[F.relu, F.relu, F.relu, None],
    H2_hidden_widths=[1500, 1024, 1024, 16 * 71],
    H2_hidden_activations=[F.relu, F.relu, F.relu, None],
    F_Gaussian=[1024, Z],
    F_Gaussian_activation=[F.relu, None],
    G1_Gaussian=[1024, H1],
    G1_Gaussian_activation=[F.relu, None],
    G2_Gaussian=[1024, H2],
    G2_Gaussian_activation=[F.relu, None]
)
n_h1=H1
n_h2=H2


model = vcca.VCCA(network_architecture, losstype1,losstype2, keepprob=dropprob,STDVAR1=stdvar1,STDVAR2=stdvar2)

Latentloss_criterion = Latentloss(n_h1, n_h2)

if H1>0:
    Reconstructloss_criterion1 = ReconstructionLoss(losstype1, stdvar1)
if H2>0:
    Reconstructloss_criterion2 = ReconstructionLoss(losstype2, stdvar2)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)

def train_model(trainData, tuneData, testData, checkpoint, batch_size=1, max_epochs=300, save_interval=1, keepprob=1.0, tune_lr=0.001):
    L = 5
    n_input1 = 39*71
    n_input2 = 16*71
    model.train()

    n_samples = trainData.num_examples
    total_batch = int(math.ceil(1.0 * n_samples / batch_size))
    epoch=0
    for index in range(epoch):
        trainData.rshuffle()

    while epoch < max_epochs:

        avg_cost=0.0
        trainData.rshuffle()
        for i in range(total_batch):
            optimizer.zero_grad()
            batch_x1, batch_x2, _, prior_zmean, prior_zvar, prior_h1_mean, prior_h1var, prior_h2_mean, prior_h2var = \
                trainData.next_batch_rshuffle(batch_size)

            z_mean, z_log_sigma_sq, h1_mean, h1_log_sigma_sq, h2_mean, h2_log_sigma_sq, \
            x1_reconstr_mean_from_z1, x1_reconstr_log_sigma_from_z1, x2_reconstr_mean_from_z2, \
            x2_reconstr_log_sigma_from_z2 = model.forward(batch_x1, batch_x2)

            cost1 = Latentloss_criterion.forward(z_mean, z_log_sigma_sq, prior_zmean, prior_zvar, h1_mean, h2_mean,
                    h1_log_sigma_sq, h2_log_sigma_sq, prior_h1_mean, prior_h2_mean, prior_h1var, prior_h2var)
            cost2 = Reconstructloss_criterion1.forward(batch_x1.repeat(L), x1_reconstr_mean_from_z1,
                                                       x1_reconstr_log_sigma_from_z1, n_input1)

            cost3 = Reconstructloss_criterion2.forward(batch_x2.repeat(L), x1_reconstr_mean_from_z1,
                                                       x1_reconstr_log_sigma_from_z1, n_input2)

            cost = cost1+cost2+cost3
            cost = cost.data[0]

            cost.backward()
            optimizer.step()

            avg_cost += cost / n_samples * batch_size

        epoch+=1
        tune_cost=0
        print("Epoch: %04d, nll1=%12.8f, nll2=%12.8f, latent loss=%12.8f, train regret=%12.8f, tune cost=%12.8f" % (
        epoch, cost1, cost2, cost3, avg_cost, tune_cost))

# Third, load the data.

trainData,tuneData,testData=read_xrmb()

train_model(trainData, tuneData, testData, checkpoint,
            batch_size=200, max_epochs=300, save_interval=1, keepprob=(1.0-dropprob), tune_lr=0.0001)







