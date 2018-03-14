import os
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

class VCCA(object):
    
    def __init__(self, architecture, losstype1, losstype2, learning_rate=0.001, l2_penalty=0.0, latent_penalty=1.0, STDVAR1=1.0, STDVAR2=1.0):
        # Save the architecture and parameters.
        self.network_architecture=architecture
        self.l2_penalty=l2_penalty
        self.learning_rate=tf.Variable(learning_rate,trainable=False)
        self.num_samples=L=5
        self.n_input1=n_input1=architecture["n_input1"]
        self.n_input2=n_input2=architecture["n_input2"]
        self.n_z=n_z=architecture["n_z"]
        self.n_h1=n_h1=architecture["n_h1"]
        self.n_h2=n_h2=architecture["n_h2"]
        self.n_HF=n_HF=architecture["n_HF"]
        self.n_HG1=n_HG1=architecture["n_HG1"]
        self.n_HG2=n_HG2=architecture["n_HG2"]
        # Trade-off parameter for KL divergence.
        self.latent_penalty=latent_penalty
        # Gaussian standard variation for the observation models of each view, only matters for losstype=2.
        self.STDVAR1=STDVAR1
        self.STDVAR2=STDVAR2
        self.bsize=1

        # Tensorflow graph inputs.
        self.x1=tf.placeholder(tf.float32, [None, self.n_input1])
        self.x2=tf.placeholder(tf.float32, [None, self.n_input2])
        width = architecture["F_Gaussian"][len(architecture["F_Gaussian"])-1]
        self.prior_z_mean=tf.placeholder(tf.float32, [None, width])
        self.prior_z_log_sigma_sq=tf.placeholder(tf.float32, [None, width])
        width = architecture["G1_Gaussian"][len(architecture["G1_Gaussian"])-1]
        self.prior_h1_mean=tf.placeholder(tf.float32, [None, width])
        self.prior_h1_log_sigma_sq=tf.placeholder(tf.float32, [None, width])
        width = architecture["G2_Gaussian"][len(architecture["G2_Gaussian"])-1]
        self.prior_h2_mean=tf.placeholder(tf.float32, [None, width])
        self.prior_h2_log_sigma_sq=tf.placeholder(tf.float32, [None, width])
        self.keepprob=tf.placeholder(tf.float32)

        # Variables to record training progress.
        self.epoch=tf.Variable(0, trainable=False)
        self.tunecost=tf.Variable(tf.zeros([1000]), trainable=False)
        self.best_value=tf.Variable(0, trainable=False)
        
        # Initialize network weights and biases.
        initializer=tf.random_uniform_initializer(-0.05, 0.05)
        
        # Use the recognition network to obtain the Gaussian distribution (mean and log-variance) of latent codes.
        print("Building view 1 recognition network F ...")
        with tf.variable_scope("F", reuse=None, initializer=initializer):
            activation=self.x1
            width=self.n_input1
            for i in range(len(architecture["F_hidden_widths"])):
                print("\tLayer %d ..." % (i+1))
                activation=tf.nn.dropout(activation, self.keepprob)
                weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["F_hidden_widths"][i]])
                biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["F_hidden_widths"][i]])
                activation=tf.add(tf.matmul(activation, weights), biases)
                if not architecture["F_hidden_activations"][i] == None:
                    activation=architecture["F_hidden_activations"][i](activation)
                width=architecture["F_hidden_widths"][i]
            mean_activation = sigma_activation = activation
            for i in range(len(architecture["F_Gaussian"])):
                print("\tLayer %d ..." % (i+1))
                with tf.variable_scope('mean'):
                    mean_activation = tf.nn.dropout(mean_activation, self.keepprob)
                    weights=tf.get_variable("F_Mean_"+str(i+1), [width, architecture["F_Gaussian"][i]])
                    biases=tf.get_variable("F_Mean_Bias"+str(i+1), [architecture["F_Gaussian"][i]])
                    mean_activation=tf.add(tf.matmul(mean_activation, weights), biases)
                    if not architecture["F_Gaussian_activation"][i] == None:
                        mean_activation=architecture["F_Gaussian_activation"][i](mean_activation)
                with tf.variable_scope('sigma'):
                    sigma_activation = tf.nn.dropout(sigma_activation, self.keepprob)
                    weights=tf.get_variable("F_Sigma_"+str(i+1), [width, architecture["F_Gaussian"][i]])
                    biases=tf.get_variable("F_Sigma_Bias_"+str(i+1), [architecture["F_Gaussian"][i]])
                    sigma_activation=tf.add(tf.matmul(sigma_activation, weights), biases)
                    if not architecture["F_Gaussian_activation"][i] == None:
                        sigma_activation=architecture["F_Gaussian_activation"][i](sigma_activation)
                width=architecture["F_Gaussian"][i]
            self.z_mean = mean_activation
            self.z_log_sigma_sq = sigma_activation

        # Private network for view 1.
        if n_h1>0:
            print("Building view 1 private network G1 ...")
            with tf.variable_scope("G1", reuse=None, initializer=initializer):
                activation=self.x1
                width=self.n_input1
                for i in range(len(architecture["G1_hidden_widths"])):
                    print("\tLayer %d ..." % (i+1))
                    activation=tf.nn.dropout(activation, self.keepprob)
                    weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["G1_hidden_widths"][i]])
                    biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["G1_hidden_widths"][i]])
                    activation=tf.add(tf.matmul(activation, weights), biases)
                    if not architecture["G1_hidden_activations"][i] == None:
                        activation=architecture["G1_hidden_activations"][i](activation)
                    width=architecture["G1_hidden_widths"][i]
                mean_activation = sigma_activation = activation
                for i in range(len(architecture["G1_Gaussian"])):
                    print("\tLayer %d ..." % (i+1))
                    with tf.variable_scope('mean'):
                        mean_activation = tf.nn.dropout(mean_activation, self.keepprob)
                        weights=tf.get_variable("G1_Mean_"+str(i+1), [width, architecture["G1_Gaussian"][i]])
                        biases=tf.get_variable("G1_Mean_Bias"+str(i+1), [architecture["G1_Gaussian"][i]])
                        mean_activation=tf.add(tf.matmul(mean_activation, weights), biases)
                        if not architecture["G1_Gaussian_activation"][i] == None:
                            mean_activation=architecture["G1_Gaussian_activation"][i](mean_activation)
                    with tf.variable_scope('sigma'):
                        sigma_activation = tf.nn.dropout(sigma_activation, self.keepprob)
                        weights=tf.get_variable("G1_Sigma_"+str(i+1), [width, architecture["G1_Gaussian"][i]])
                        biases=tf.get_variable("G1_Sigma_Bias_"+str(i+1), [architecture["G1_Gaussian"][i]])
                        sigma_activation=tf.add(tf.matmul(sigma_activation, weights), biases)
                        if not architecture["G1_Gaussian_activation"][i] == None:
                            sigma_activation=architecture["G1_Gaussian_activation"][i](sigma_activation)
                    width=architecture["G1_Gaussian"][i]
                self.h1_mean = mean_activation
                self.h1_log_sigma_sq = sigma_activation
        
        # Private network for view 2.
        if n_h2>0:
            print("Building view 2 private network G2 ...")
            with tf.variable_scope("G2", reuse=None, initializer=initializer):
                activation=self.x2
                width=self.n_input2
                for i in range(len(architecture["G2_hidden_widths"])):
                    print("\tLayer %d ..." % (i+1))
                    activation=tf.nn.dropout(activation, self.keepprob)
                    weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["G2_hidden_widths"][i]])
                    biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["G2_hidden_widths"][i]])
                    activation=tf.add(tf.matmul(activation, weights), biases)
                    if not architecture["G2_hidden_activations"][i] == None:
                        activation=architecture["G2_hidden_activations"][i](activation)
                    width=architecture["G2_hidden_widths"][i]
                mean_activation = sigma_activation = activation
                for i in range(len(architecture["G2_Gaussian"])):
                    print("\tLayer %d ..." % (i+1))
                    with tf.variable_scope('mean'):
                        mean_activation = tf.nn.dropout(mean_activation, self.keepprob)
                        weights=tf.get_variable("G2_Mean_"+str(i+1), [width, architecture["G2_Gaussian"][i]])
                        biases=tf.get_variable("G2_Mean_Bias"+str(i+1), [architecture["G2_Gaussian"][i]])
                        mean_activation=tf.add(tf.matmul(mean_activation, weights), biases)
                        if not architecture["G2_Gaussian_activation"][i] == None:
                            mean_activation=architecture["G2_Gaussian_activation"][i](mean_activation)
                    with tf.variable_scope('sigma'):
                        sigma_activation = tf.nn.dropout(sigma_activation, self.keepprob)
                        weights=tf.get_variable("G2_Sigma_"+str(i+1), [width, architecture["G2_Gaussian"][i]])
                        biases=tf.get_variable("G2_Sigma_Bias_"+str(i+1), [architecture["G2_Gaussian"][i]])
                        sigma_activation=tf.add(tf.matmul(sigma_activation, weights), biases)
                        if not architecture["G2_Gaussian_activation"][i] == None:
                            sigma_activation=architecture["G2_Gaussian_activation"][i](sigma_activation)
                    width=architecture["G2_Gaussian"][i]
                self.h2_mean = mean_activation
                self.h2_log_sigma_sq = sigma_activation
        
        # Calculate latent losses (KL divergence) for shared and private variables.
        subz = self.z_mean - self.prior_z_mean
        logz = tf.log(self.prior_z_log_sigma_sq)*2
        expz = tf.exp(logz)
        latent_loss_z = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - logz - tf.square(subz)/expz - tf.exp(self.z_log_sigma_sq)/expz, 1)
        
        if n_h1>0:
            subh1 = self.h1_mean - self.prior_h1_mean
            logh1 = tf.log(self.prior_h1_log_sigma_sq)*2
            exph1 = tf.exp(logh1)
            latent_loss_h1 = -0.5 * tf.reduce_sum(1 + self.h1_log_sigma_sq - logh1 - tf.square(subh1)/exph1 - tf.exp(self.h1_log_sigma_sq)/exph1, 1)
        else:
            latent_loss_h1=tf.constant(0.0)
        
        if n_h2>0:
            subh2 = self.h2_mean - self.prior_h2_mean
            logh2 = tf.log(self.prior_h2_log_sigma_sq)*2
            exph2 = tf.exp(logh2)
            latent_loss_h2 = -0.5 * tf.reduce_sum(1 + self.h2_log_sigma_sq - logh2 - tf.square(subh2)/exph2 - tf.exp(self.h2_log_sigma_sq)/exph2, 1)
        else:
            latent_loss_h2=tf.constant(0.0)
        self.latent_loss=tf.reduce_mean(latent_loss_z + latent_loss_h1 + latent_loss_h2)
        
        # Draw L samples of z.
        z_epsshape=tf.mul(tf.shape(self.z_mean), [L,1])
        eps=tf.random_normal(z_epsshape, 0, 1, dtype=tf.float32)
        self.z1=tf.add( tf.tile(self.z_mean, [L,1]), tf.mul( tf.tile(tf.exp(0.5 * self.z_log_sigma_sq), [L,1]), eps))
        
        # Draw L samples of h1.
        if n_h1>0:
            h1_epsshape=tf.mul(tf.shape(self.h1_mean), [L,1])
            eps=tf.random_normal(h1_epsshape, 0, 1, dtype=tf.float32)
            self.h1=tf.add( tf.tile(self.h1_mean, [L,1]), tf.mul( tf.tile(tf.exp(0.5 * self.h1_log_sigma_sq), [L,1]), eps))
        
        # Use the generator network to reconstruct view 1.
        print("Building view 1 reconstruction network H1 ...")
        if n_h1>0:
            activation=tf.concat(1, [self.z1, self.h1])
            width=n_z + n_h1
        else:
            activation=self.z1
            width=n_z
            
        with tf.variable_scope("H1", reuse=None, initializer=initializer):
            for i in range(len(architecture["H1_hidden_widths"])):
                print("\tLayer %d ..." % (i+1))
                activation=tf.nn.dropout(activation, self.keepprob)
                if i==(len(architecture["H1_hidden_widths"])-1):
                    weights=tf.get_variable("weights_log_sigma_sq", [width, architecture["H1_hidden_widths"][i]])
                    biases=tf.get_variable("biases_log_sigma_sq", [architecture["H1_hidden_widths"][i]])
                    self.x1_reconstr_log_sigma_sq_from_z1=tf.add(tf.matmul(activation, weights), biases)
                weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["H1_hidden_widths"][i]])
                biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["H1_hidden_widths"][i]])
                activation=tf.add(tf.matmul(activation, weights), biases)
                if not architecture["H1_hidden_activations"][i] == None:
                    activation=architecture["H1_hidden_activations"][i](activation)
                width=architecture["H1_hidden_widths"][i]
        self.x1_reconstr_mean_from_z1=activation
                
        # Draw L samples of z.
        eps=tf.random_normal(z_epsshape, 0, 1, dtype=tf.float32)
        self.z2=tf.add( tf.tile(self.z_mean, [L,1]), tf.mul( tf.tile(tf.exp(0.5 * self.z_log_sigma_sq), [L,1]), eps))

        # Draw L samples of h2.
        if n_h2>0:
            h2_epsshape=tf.mul(tf.shape(self.h2_mean), [L,1])
            eps=tf.random_normal(h2_epsshape, 0, 1, dtype=tf.float32)
            self.h2=tf.add( tf.tile(self.h2_mean, [L,1]), tf.mul( tf.tile(tf.exp(0.5 * self.h2_log_sigma_sq), [L,1]), eps))

        # Use the generator network to reconstruct view 2.
        print("Building view 2 reconstruction network H2 ...")
        if n_h2>0:
            activation=tf.concat(1, [self.z2, self.h2])
            width=n_z + n_h2
        else:
            activation=self.z2
            width=n_z

        with tf.variable_scope("H2", reuse=None, initializer=initializer):
            for i in range(len(architecture["H2_hidden_widths"])):
                print("\tLayer %d ..." % (i+1))
                activation=tf.nn.dropout(activation, self.keepprob)
                if i==(len(architecture["H2_hidden_widths"])-1):
                    weights=tf.get_variable("weights_log_sigma_sq", [width, architecture["H2_hidden_widths"][i]])
                    biases=tf.get_variable("biases_log_sigma_sq", [architecture["H2_hidden_widths"][i]])
                    self.x2_reconstr_log_sigma_sq_from_z2=tf.add(tf.matmul(activation, weights), biases)
                weights=tf.get_variable("weights_layer_" + str(i+1), [width, architecture["H2_hidden_widths"][i]])
                biases=tf.get_variable("biases_layer_" + str(i+1), [architecture["H2_hidden_widths"][i]])
                activation=tf.add(tf.matmul(activation, weights), biases)
                if not architecture["H2_hidden_activations"][i] == None:
                    activation=architecture["H2_hidden_activations"][i](activation)
                width=architecture["H2_hidden_widths"][i]
        self.x2_reconstr_mean_from_z2=activation

        
        # Compute negative log-likelihood for input data.
        self.nll1=self._compute_reconstr_loss( tf.tile(self.x1, [L,1]), self.x1_reconstr_mean_from_z1, self.x1_reconstr_log_sigma_sq_from_z1, self.n_input1, losstype1, STDVAR1)

        self.nll2=self._compute_reconstr_loss( tf.tile(self.x2, [L,1]), self.x2_reconstr_mean_from_z2, self.x2_reconstr_log_sigma_sq_from_z2, self.n_input2, losstype2, STDVAR2)

        
        # Weight decay.
        self.weightdecay=tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        
        # Define cost and use the ADAM optimizer.
        self.cost=latent_penalty * self.latent_loss + self.nll1 + self.nll2 + \
                   l2_penalty * self.weightdecay
        self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        
        # Initializing the tensor flow variables and launch the session.
        NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
        init=tf.initialize_all_variables()
        self.sess=tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))
        self.sess.run(init)

    def assign_lr(self, lr):
        self.sess.run(tf.assign(self.learning_rate, lr))

    def assign_epoch(self, EPOCH_VALUE):
        self.sess.run(tf.assign(self.epoch, EPOCH_VALUE))
    
    def assign_tunecost(self, TUNECOST_VALUE):
        self.sess.run(tf.assign(self.tunecost, TUNECOST_VALUE))

    def assign_bestvalue(self, BEST_VALUE):
        self.sess.run(tf.assign(self.best_value, BEST_VALUE))

    def sample_bernoulli(self, MEAN):
        mshape=tf.shape(MEAN)
        return tf.select( tf.random_uniform(mshape) < MEAN, tf.ones(mshape), tf.zeros(mshape))

    
    def _compute_reconstr_loss(self, x_input, x_reconstr_mean, x_reconstr_log_sigma_sq, n_out, losstype, STDVAR):
        
        if losstype==0:
            # Cross entropy loss.
            reconstr_loss=- tf.reduce_sum(x_input * tf.log( 1e-6 + x_reconstr_mean ) + ( 1 - x_input ) * tf.log( 1e-6 + 1 - x_reconstr_mean), 1)
        elif losstype==1:
            # Least squares loss, with learned std.
            reconstr_loss=0.5 * tf.reduce_sum( tf.div( tf.square(x_input - x_reconstr_mean), 1e-6 + tf.exp(x_reconstr_log_sigma_sq) ), 1 ) + 0.5 * tf.reduce_sum( x_reconstr_log_sigma_sq, 1 ) + 0.5 * math.log(2 * math.pi) * n_out
        elif losstype==2:
            # Least squares loss, with specified std.
            reconstr_loss=0.5 * tf.reduce_sum( tf.square( (x_input - x_reconstr_mean)/STDVAR ), 1 ) + 0.5 * math.log(2 * math.pi * STDVAR * STDVAR) * n_out
        
        # Average over the minibatch.
        cost=tf.reduce_mean(reconstr_loss)
        return cost
    
    
    def partial_fit(self, X1, X2, zmean, zvar, h1mean, h1var, h2mean, h2var, keepprob):
        # Train model based on mini-batch of input data. Return cost of mini-batch.
        opt, cost1, cost2, cost3 = self.sess.run( [self.optimizer, self.nll1, self.nll2, self.latent_loss], feed_dict={self.x1: X1, self.x2: X2, self.prior_z_mean: zmean, self.prior_z_log_sigma_sq: zvar, self.prior_h1_mean: h1mean, self.prior_h1_log_sigma_sq: h1var, self.prior_h2_mean: h2mean, self.prior_h2_log_sigma_sq: h2var, self.keepprob: keepprob})
        return cost1, cost2, cost3
    

    def transform_shared_minibatch(self, view, X):
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution.
        if view==1:
            return self.sess.run( [self.z_mean, tf.exp(0.5 * self.z_log_sigma_sq)], feed_dict={self.x1: X, self.keepprob: 1.0})
        else:
            raise ValueError("The shared variable is extracted from view 1!")    
    
    
    def transform_shared(self, view, X):
        
        N=X.shape[0]
        Din=X.shape[1]
        xtmp,_=self.transform_shared_minibatch(view, X[np.newaxis,0,:])
        Dout=xtmp.shape[1]
        
        Ymean=np.zeros([N, Dout], dtype=np.float32)
        Ystd=np.zeros([N, Dout], dtype=np.float32)
        batchsize=5000
        for batchidx in range(1+np.ceil(N / batchsize).astype(int)):
            idx=range( batchidx*batchsize, min(N, (batchidx+1)*batchsize) )
            tmpmean, tmpstd=self.transform_shared_minibatch(view, X[idx,:])
            Ymean[idx,:]=tmpmean
            Ystd[idx,:]=tmpstd
        return Ymean, Ystd
    
    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.
        
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent 
        space.        
        """
        # Note: This maps to mean of distribution, we could alternatively sample from Gaussian distribution.
        if z_mu is None:
            z_mu=np.random.normal(size=self.n_z)

        # It does not matter whether we use self.z1 or self.z2 below as the generation networks are shared.
        x1_recon=dict()
        x1_recon["mean"]=self.sess.run( self.x1_reconstr_mean_from_z1, feed_dict={self.z1: z_mu, self.keepprob: 1.0})
        x1_recon["std"]=self.sess.run( tf.exp(0.5 * self.x1_reconstr_log_sigma_sq_from_z1), feed_dict={self.z1: z_mu, self.keepprob: 1.0})
        x2_recon=dict()
        x2_recon["mean"]=self.sess.run( self.x2_reconstr_mean_from_z2, feed_dict={self.z1: z_mu, self.keepprob: 1.0})
        x2_recon["std"]=self.sess.run( tf.exp(0.5 * self.x2_reconstr_log_sigma_sq_from_z2), feed_dict={self.z1: z_mu, self.keepprob: 1.0})

        return x1_recon, x2_recon
    
    
    def reconstruct(self, view, X1, X2):
        """ Use VCCA to reconstruct given data. """
        if view==1:
            x1_recon_mean, x1_recon_std=self.sess.run( [self.x1_reconstr_mean_from_z1, tf.exp(0.5 * self.x1_reconstr_log_sigma_sq_from_z1)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})
            x2_recon_mean, x2_recon_std=self.sess.run( [self.x2_reconstr_mean_from_z2, tf.exp(0.5 * self.x2_reconstr_log_sigma_sq_from_z2)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})
        else:
            x1_recon_mean, x1_recon_std=self.sess.run( [self.x1_reconstr_mean_from_z1, tf.exp(0.5 * self.x1_reconstr_log_sigma_sq_from_z1)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})
            x2_recon_mean, x2_recon_std=self.sess.run( [self.x2_reconstr_mean_from_z2, tf.exp(0.5 * self.x2_reconstr_log_sigma_sq_from_z2)], feed_dict={self.x1: X1, self.x2: X2, self.keepprob: 1.0})

        x1_recon=dict();  x1_recon["mean"]=x1_recon_mean;  x1_recon["std"]=x1_recon_std
        x2_recon=dict();  x2_recon["mean"]=x2_recon_mean;  x2_recon["std"]=x2_recon_std
        
        return x1_recon, x2_recon
    
    
def train(model, trainData, tuneData, testData, saver, checkpoint, batch_size=1, max_epochs=300, save_interval=1, keepprob=1.0, tune_lr=0.001):
    epoch=model.sess.run(model.epoch)
    lr=model.sess.run(model.learning_rate)
    lr=tune_lr
    model.assign_lr(lr)
    n_samples=trainData.num_examples
    total_batch=int(math.ceil(1.0 * n_samples / batch_size))

    for index in range(epoch):
        trainData.rshuffle()

    # Training cycle.
    while epoch < max_epochs:
        print("Current learning rate %f" % lr)
        avg_cost=0.0
        trainData.rshuffle()
        # Loop over all batches.
        for i in range(total_batch):
            batch_x1, batch_x2, _, zmean, zvar, h1mean, h1var, h2mean, h2var = trainData.next_batch_rshuffle(batch_size)
            cost1, cost2, cost3 = model.partial_fit(batch_x1, batch_x2, zmean, zvar, h1mean, h1var, h2mean, h2var, keepprob)
            cost = cost1+cost2+cost3
            avg_cost += cost / n_samples*batch_size
        # Display logs per epoch step.
        epoch=epoch+1
        tune_cost=0
        print("Epoch: %04d, nll1=%12.8f, nll2=%12.8f, latent loss=%12.8f, train regret=%12.8f, tune cost=%12.8f" % (epoch, cost1, cost2, cost3, avg_cost, tune_cost))
        if (checkpoint) and (epoch % save_interval == 0):
            model.assign_epoch(epoch)
            save_path=saver.save(model.sess, checkpoint)
            print("Model saved in file: %s" % save_path)

        if (epoch % 15 == 0):
            import scipy.io as sio
            recogfeat, _ = model.transform_shared(1,testData._images1)
            sio.savemat(str(epoch)+"_"+checkpoint, { "feat":recogfeat })

    return model
