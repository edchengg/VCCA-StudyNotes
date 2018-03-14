import numpy as np
import math
import os
import tensorflow as tf
import vcca_IM as vcca
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


def visualize(XV,tuneLabel):

    vowels=np.array([2, 4, 25, 34, 18, 1, 11])-1
    tags=['AE', 'AO', 'OW', 'UW', 'IY', 'AA', 'EH']
    # tuneLabel=np.reshape(tuneLabel,[len(tuneLabel)])
    
    # Obtain counts and indices.
    idx=[]
    for i in range(len(vowels)):
        idx1=np.argwhere(tuneLabel==vowels[i])
        idx+=[idx1[j,0] for j in range(len(idx1))]
    
    input=XV[idx,:]
    input=input[0:len(idx):24,:]
    lab=tuneLabel[idx]
    lab=lab[0:len(idx):24]
    
    # Fit t-SNE.
    from sklearn.manifold import TSNE
    tsne=TSNE(perplexity=20, n_components=2, init="pca", n_iter=5000)
    emb=tsne.fit_transform( np.asfarray(input, dtype="float") )

    return emb, lab


def main(argv=None):

    # Set random seeds.
    np.random.seed(0)
    tf.set_random_seed(0)
    
    # Obtain parsed arguments.
    Z=args.Z
    print("Dimensionality of shared variables: %d" % Z)
    H1=args.H1
    print("Dimensionality of view 1 private variables: %d" % H1)
    H2=args.H2
    print("Dimensionality of view 2 private variables: %d" % H2)
    HF=args.HF
    HG1=args.HG1
    HG2=args.HG2
    IM_penalty=args.IM
    print("Regularization constant for IM penalty: %f" % IM_penalty)
    dropprob=args.dropprob
    print("Dropout rate: %f" % dropprob)
    stdvar1=args.stdvar1
    print("View 1 observation std: %f" % stdvar1)
    stdvar2=args.stdvar2
    print("View 2 observation std: %f" % stdvar2)
    latent_penalty=args.zpenalty
    print("Latent penalty: %f" % latent_penalty)
    checkpoint=args.checkpoint
    print("Trained model will be saved at %s" % checkpoint)
    
    # Some configurations.
    losstype1=2        # Gaussian with given stdvar1.
    losstype2=2        # Gaussian with given stdvar2.
    learning_rate=0.0001
    l2_penalty=0.0
    
    # File for saving classification results.
    classfile=checkpoint + '_features.mat'
    if os.path.isfile(classfile):
        print("Job is already finished!")
        return
    

    # Define network architectures.
    network_architecture=dict(
        n_input1=39*71, # XRMB data MFCCs input 
        n_input2=16*71, # XRMB data articulation input 
        n_z=Z,  # Dimensionality of shared latent space
        n_h1=H1, # Dimensionality of individual latent space of view 1
        n_h2=H2, # Dimensionality of individual latent space of view 2
        n_HF=HF, # Dimensionality of LSTM-F
        n_HG1=HG1, # Dimensionality of LSTM-G1
        n_HG2=HG2, # Dimensionality of LSTM-G2
        F_hidden_widths=[1024, 1024],
        F_hidden_activations=[tf.nn.relu, tf.nn.relu, None],
        G1_hidden_widths=[1024, 1024],
        G1_hidden_activations=[tf.nn.relu, tf.nn.relu, None],
        G2_hidden_widths=[1024, 1024],
        G2_hidden_activations=[tf.nn.relu, tf.nn.relu, None],
        H1_hidden_widths=[1500, 1024, 1024, 39*71],
        H1_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
        H2_hidden_widths=[1500, 1024, 1024, 16*71],
        H2_hidden_activations=[tf.nn.relu, tf.nn.relu, tf.nn.relu, None],
        F_Gaussian=[1024, Z],
        F_Gaussian_activation=[tf.nn.relu, None],
        G1_Gaussian=[1024, H1],
        G1_Gaussian_activation=[tf.nn.relu, None],
        G2_Gaussian=[1024, H2],
        G2_Gaussian_activation=[tf.nn.relu, None]
        )
    
    # First, build the model.
    model=vcca.VCCA(network_architecture, losstype1, losstype2, learning_rate, l2_penalty, latent_penalty, stdvar1, stdvar2)
    saver=tf.train.Saver()

    # Second, load the saved moded, if provided.
    if checkpoint and os.path.isfile(checkpoint):
        print("loading model from %s " % checkpoint)
        saver.restore(model.sess, checkpoint)
        epoch=model.sess.run(model.epoch)
        print("picking up from epoch %d " % epoch)
    else:
        print("checkpoint file not given or not existent!")
        epoch = 0

    np.random.seed(epoch)
    tf.set_random_seed(epoch)
    
    # Third, load the data.
    with tf.device('/cpu:0'):
        trainData,tuneData,testData=read_xrmb()
    
    # Traning.
    model=vcca.train(model, trainData, tuneData, testData, saver, checkpoint, batch_size=200, max_epochs=300, save_interval=1, keepprob=(1.0-dropprob), tune_lr=0.0001)

    # Map recognition data.
    import scipy.io as sio
    recogfeat, _ = model.transform_shared(1,testData._images1)
    recogfeat2, _ = model.transform_shared(1,trainData._images1)
    recogfeat3, _ = model.transform_shared(1,tuneData._images1)

    sio.savemat(classfile, { "feat":np.concatenate((recogfeat2,recogfeat3,recogfeat),0) })


if __name__ == "__main__":
    tf.app.run()


