import numpy as np
import math
import os
import tensorflow as tf
import ctc as ctc
from myreadinput_mfcc import read_xrmb


import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="./ctc_xrmb", help="Path to saved models", type=str) 
parser.add_argument("--bestcheckpoint", default="./best_ctc_xrmb", help="Path to saved best models", type=str) 
parser.add_argument("--lr", default=0.0001, help="Learning rate", type=float)
parser.add_argument("--num_hidden", default=64, help="Num of units of LSTM", type=int)
parser.add_argument("--num_classes", default=41, help="Num of classes", type=int)
parser.add_argument("--num_layers", default=2, help="Num of layers", type=int)
parser.add_argument("--n_input", default=100, help="Num of features", type=int)
parser.add_argument("--batch_size", default=5, help="NUm of sequences", type=int)
parser.add_argument("--dropprob", default=0.0, help="Drop out rate", type=float)
parser.add_argument("--fold", default=0, help="Fold", type=int)
args=parser.parse_args()

def main(argv=None):

    # Set random seeds.
    np.random.seed(0)
    tf.set_random_seed(0)

    # Obtain parsed arguments.
    learning_rate=args.lr
    print("Learning rate will be %f" %learning_rate)
    checkpoint=args.checkpoint
    print("Trained model will be saved at %s" % checkpoint)
    bestcheckpoint=args.bestcheckpoint
    print("Best model will be saved at %s" % bestcheckpoint)
    num_hidden=args.num_hidden
    print("Num of hidden units will be %d" % num_hidden)
    num_classes=args.num_classes
    print("Num of classes is %d" % num_classes)
    num_layers=args.num_layers
    print("Num of layers is %d" % num_layers)
    n_input=args.n_input
    print("Num of features is %d" % n_input)
    batch_size=args.batch_size
    print("Batch size is %d" % batch_size)
    dropprob=args.dropprob
    print("Drop prob is %f" % dropprob)
    fold=args.fold
    print("Fold is %d" % fold)

    # First, build the model.
    model=ctc.CTC(learning_rate, num_classes, num_hidden, num_layers, n_input, batch_size)
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

    # Third, load the data.
    with tf.device('/cpu:0'):
        trainData, tuneData, testData=read_xrmb(fold)
    avg_err = 0.0
    for ifold in range(30):
        model=ctc.train(model, trainData, tuneData, saver, checkpoint, bestcheckpoint, batch_size, max_epochs=(ifold+1), keepprob=(1.0-dropprob), tune_lr=0.0005)
        # Fifth, test
        err=model.sess.run(model.besttune)
        cost=model.sess.run(model.mycost)
        print err
        print cost
        if (cost<1):
            break

    saver.restore(model.sess, bestcheckpoint)
    err = model.evaluate_cost(testData)
    print("Final Error is %12.8f" % err)

if __name__ == "__main__":
    tf.app.run()


