import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
#from tensorflow.python.ops import ctc_ops
from tensorflow.contrib.ctc import ctc_ops
from utils import sparse_tuple_from as sparse_tuple_from
import os

class CTC(object):    
    def __init__(self, learning_rate=0.0001, num_classes=42, num_hidden=64, num_layers=1, n_input=100, batch_size=1):

        # Save the architecture and parameters.
        self.learning_rate=tf.Variable(learning_rate,trainable=False)
        self.num_classes=num_classes
        self.num_hidden=num_hidden
        self.num_layers=num_layers
        self.n_input=n_input
        self.bsize=batch_size

        # Tensorflow graph inputs.
        self.x=tf.placeholder(tf.float32, [None, None, self.n_input])
        self.targets = tf.sparse_placeholder(tf.int32)
        self.mylen=tf.placeholder(tf.int32, [None])
        self.mylen64=tf.to_int64(self.mylen)
        self.keepprob=tf.placeholder(tf.float32)
        self.epoch=tf.Variable(0, trainable=False)
        self.mycost=tf.Variable(0, trainable=False)
        self.besttune=tf.Variable(999999, trainable=False)

        initializer=tf.random_uniform_initializer(-0.05, 0.05)

        proj_shared_gru_fw_cell_1st = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        proj_shared_gru_bw_cell_1st = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        proj_shared_gru_fw_cell_2nd = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        proj_shared_gru_bw_cell_2nd = tf.nn.rnn_cell.LSTMCell(self.num_hidden)
        
        activation = self.x
        inv_activation = tf.reverse_sequence(activation, self.mylen64, 1)
        with tf.variable_scope('l2r_layer1_shared_projection'):
            output1, _ = tf.nn.dynamic_rnn(proj_shared_gru_fw_cell_1st, activation, dtype=tf.float32, sequence_length=self.mylen)
        with tf.variable_scope('r2l_layer1_shared_projection'):
            output2, _ = tf.nn.dynamic_rnn(proj_shared_gru_bw_cell_1st, inv_activation, dtype=tf.float32, sequence_length=self.mylen)
        
        output2 = tf.reverse_sequence(output2, self.mylen64, 1)
        activation = tf.nn.dropout(tf.concat(2, [output1, output2]), self.keepprob)
        inv_activation = tf.reverse_sequence(activation, self.mylen64, 1)
        with tf.variable_scope('l2r_layer2_shared_projection'):
            output1, _ = tf.nn.dynamic_rnn(proj_shared_gru_fw_cell_2nd, activation, dtype=tf.float32, sequence_length=self.mylen)
        with tf.variable_scope('r2l_layer2_shared_projection'):
            output2, _ = tf.nn.dynamic_rnn(proj_shared_gru_bw_cell_2nd, inv_activation, dtype=tf.float32, sequence_length=self.mylen)
        
        output2 = tf.reverse_sequence(output2, self.mylen64, 1)
        outputs = tf.concat(2, [output1, output2])
        outputs = tf.reshape(outputs, [-1, 2*self.num_hidden])

        with tf.variable_scope("W", reuse=None, initializer=initializer):
            W = tf.get_variable("WP", [2*self.num_hidden,self.num_classes])
            b = tf.get_variable("BP", [self.num_classes])
            logits = tf.add(tf.matmul(outputs, W),b)

        # Reshaping back to the original shape
        logits = tf.reshape(logits, [self.bsize, -1, self.num_classes])
        # Time major
        logits = tf.transpose(logits, (1, 0, 2))

        loss = ctc_ops.ctc_loss(logits, self.targets, self.mylen)
        self.cost = tf.reduce_mean(loss)
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.decoded, self.log_prob = ctc_ops.ctc_beam_search_decoder(logits, self.mylen, beam_width=100, top_paths=100, merge_repeated=True)
        self.ler= tf.reduce_sum(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets, normalize=True))

        NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
        NUM_THREADS = 1
        init=tf.initialize_all_variables()
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS))
        self.sess.run(init)

    def assign_cost(self, value):
        self.sess.run(tf.assign(self.mycost, value))

    def assign_tunecost(self, value):
        self.sess.run(tf.assign(self.besttune, value))

    def assign_lr(self, lr):
        self.sess.run(tf.assign(self.learning_rate, lr))

    def assign_epoch(self, EPOCH_VALUE):
        self.sess.run(tf.assign(self.epoch, EPOCH_VALUE))
    
    def partial_fit(self, X, targets, tmplen, keepprob):
        opt, cost=self.sess.run( [self.optimizer, self.cost], feed_dict={self.x:X, self.targets:targets, self.mylen:tmplen, self.keepprob:keepprob})
        return cost
    
    def evaluate_cost(self, X):
        NN = (X.lengths).shape[0]
        N = (X.images).shape[0]
        avg_cost=0.0
        start = 0
        total = 0
        total_batch=int(math.ceil(1.0 * NN / self.bsize))
        for batchidx in range(total_batch):
            batch_x, labels, tmplen, mysize=X.next_batch(self.bsize, start)
            # Need to convert labels to targets
            test_targets = sparse_tuple_from(labels)
            error,A=self.sess.run( [self.ler,self.decoded[0]], feed_dict={self.x: batch_x, self.targets: test_targets, self.mylen:tmplen, self.keepprob: 1.0})
            print A.values
            print test_targets[1]
            avg_cost += error
            start += self.bsize
        return avg_cost/NN



def train(model, trainData, tuneData, saver, checkpoint, bestcheckpoint, batch_size=1, max_epochs=300, keepprob=1.0, tune_lr=0.0005):

    epoch=model.sess.run(model.epoch)
    lr=model.sess.run(model.learning_rate)
    lr=tune_lr
    model.assign_lr(lr)
    n_seqs=trainData.num_seqs
    n_samples=trainData.num_examples
    total_batch = n_seqs/batch_size

    for index in range(epoch+1):
        trainData.rshuffle()

    while epoch < max_epochs:
        print("Current learning rate %f" % lr)
        avg_cost=0.0
        for index in range(total_batch):
            batch_x, labels, tmplen, mysize=trainData.next_batch(batch_size, index)
            train_targets = sparse_tuple_from(labels)
            cost=model.partial_fit(batch_x, train_targets, tmplen, keepprob)
            avg_cost += cost*mysize/n_samples

        epoch=epoch+1
        model.assign_epoch(epoch)
        model.assign_cost(avg_cost)
        trainData.rshuffle() # Shuffle data after one epoch
        save_path=saver.save(model.sess, checkpoint)
        print("Model saved in file: %s" % save_path)
        print("Epoch: %04d, total train regret=%12.8f" % (epoch, avg_cost))
        btune = model.sess.run(model.besttune)
        tunecost = 1000*model.evaluate_cost(tuneData)
        print("Epoch: %04d, total tune regret=%12.8f" % (epoch, tunecost))
        if (tunecost<btune):
            model.assign_tunecost(tunecost)
            save_path=saver.save(model.sess, bestcheckpoint)

    return model
