import scipy.io as sio
import tensorflow as tf
import numpy as np
import torch


class DataSet(object):
    
    def __init__(self, images1, images2, labels, zmean, zvar, h1mean, h1var, h2mean, h2var, fake_data=False, one_hot=False, dtype=tf.float32):
        """Construct a DataSet.
        """
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        
        if fake_data:
            self._num_examples = 10000
            self._num_seqs = 7
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], ('images1.shape: %s labels.shape: %s' % (images1.shape,labels.shape))
            assert images2.shape[0] == labels.shape[0], ('images2.shape: %s labels.shape: %s' % (images2.shape,labels.shape))
            # _num_examples is the total frame
            self._num_examples = images1.shape[0]
            # _num_seqs is the total sequences
            if dtype == tf.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                print("type conversion view 1")
                images1 = images1.astype(np.float32)
            if dtype == tf.float32 and images2.dtype != np.float32:
                print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._zmean = zmean
        self._h1mean = h1mean
        self._h2mean = h2mean
        self._zvar = zvar
        self._h1var = h1var
        self._h2var = h2var
        self._index_in_epoch = 0
        self._parray = np.arange(self._num_examples)

    @property
    def images1(self):
        return self._images1
    
    @property
    def images2(self):
        return self._images2
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def rshuffle(self):
        np.random.shuffle(self._parray)

    def next_batch_rshuffle(self, batch_size, fake_data=False):
        v1 = self._images1[-1:-1]
        v2 = self._images2[-1:-1]
        l = self._labels[-1:-1]
        zm = self._zmean[-1:-1]
        zv = self._zvar[-1:-1]
        h1m = self._h1mean[-1:-1]
        h1v = self._h1var[-1:-1]
        h2m = self._h2mean[-1:-1]
        h2v = self._h2var[-1:-1]

        for index in range(batch_size):
            self._index_in_epoch %= self._num_examples
            pos = self._parray[self._index_in_epoch]
            v1 = np.concatenate((v1,self._images1[pos:pos+1]))
            v2 = np.concatenate((v2,self._images2[pos:pos+1]))
            l = np.concatenate((l,self._labels[pos:pos+1]))
            zm = np.concatenate((zm,self._zmean[pos:pos+1]))
            zv = np.concatenate((zv,self._zvar[pos:pos+1]))
            h1m = np.concatenate((h1m,self._h1mean[pos:pos+1]))
            h1v = np.concatenate((h1v,self._h1var[pos:pos+1]))
            h2m = np.concatenate((h2m,self._h2mean[pos:pos+1]))
            h2v = np.concatenate((h2v,self._h2var[pos:pos+1]))
            self._index_in_epoch += 1
        return v1, v2, l, zm, zv, h1m, h1v, h2m, h2v

def read_xrmb():
    train_prior_z = sio.loadmat('./train_z.mat')
    train_prior_h1 = sio.loadmat('./train_h1.mat')
    train_prior_h2 = sio.loadmat('./train_h2.mat')
    #test_prior_z = sio.loadmat('./test_z.mat')
    #test_prior_h1 = sio.loadmat('./test_h1.mat')
    #test_prior_h2 = sio.loadmat('./test_h2.mat')
    #tune_prior_z = sio.loadmat('./tune_z.mat')
    #tune_prior_h1 = sio.loadmat('./tune_h1.mat')
    #tune_prior_h2 = sio.loadmat('./tune_h2.mat')
    train_z_mean = train_prior_z['mean']
    train_z_var = train_prior_z['var']
    train_h1_mean = train_prior_h1['mean']
    train_h1_var = train_prior_h1['var']
    train_h2_mean = train_prior_h2['mean']
    train_h2_var = train_prior_h2['var']
    #tune_z_mean = tune_prior_z['mean']
    #tune_z_var = tune_prior_z['var']
    #tune_h1_mean = tune_prior_h1['mean']
    #tune_h1_var = tune_prior_h1['var']
    #tune_h2_mean = tune_prior_h2['mean']
    #tune_h2_var = tune_prior_h2['var']
    #test_z_mean = test_prior_z['mean']
    #test_z_var = test_prior_z['var']
    #test_h1_mean = test_prior_h1['mean']
    #test_h1_var = test_prior_h1['var']
    #test_h2_mean = test_prior_h2['mean']
    #test_h2_var = test_prior_h2['var']

    data=sio.loadmat('./XRMB_SEQ.mat')
    VIEW1 = data['MFCCS']
    VIEW2 = data['ARTICS']
    LENGTH = data['LENGTHS']
    LABEL = data['LABELS']

    CUT1 = 1500
    CUT2 = 1736
    LEN = 2357
    TOTAL_FRAME = 2430668

    num1 = 0
    num2 = 0
    for index in range(CUT1):
        num1 += LENGTH[index][0]
    for index in range(CUT1,CUT2):
        num2 += LENGTH[index][0]

    VIEW1_VALIDATE = VIEW1[num1:num1+num2]
    VIEW1_TRAIN = VIEW1[0:num1]
    VIEW1_TEST = VIEW1[num1+num2:TOTAL_FRAME]

    VIEW2_VALIDATE = VIEW2[num1:num1+num2]
    VIEW2_TRAIN = VIEW2[0:num1]
    VIEW2_TEST = VIEW2[num1+num2:TOTAL_FRAME]

    LABEL_VALIDATE = LABEL[num1:num1+num2]
    LABEL_TRAIN = LABEL[0:num1]
    LABEL_TEST = LABEL[num1+num2:TOTAL_FRAME]

    LENGTH_TRAIN = LENGTH[0:CUT1]
    LENGTH_VALIDATE = LENGTH[CUT1:CUT2]
    LENGTH_TEST = LENGTH[CUT2:LEN]

    NUM = 71
    HALF = 35
    VVIEW1_TRAIN = np.zeros((VIEW1_TRAIN.shape[0], VIEW1_TRAIN.shape[1]*NUM))
    VVIEW2_TRAIN = np.zeros((VIEW2_TRAIN.shape[0], VIEW2_TRAIN.shape[1]*NUM))
    VVIEW1_VALIDATE = np.zeros((VIEW1_VALIDATE.shape[0], VIEW1_VALIDATE.shape[1]*NUM))
    VVIEW2_VALIDATE = np.zeros((VIEW2_VALIDATE.shape[0], VIEW2_VALIDATE.shape[1]*NUM))
    VVIEW1_TEST = np.zeros((VIEW1_TEST.shape[0], VIEW1_TEST.shape[1]*NUM))
    VVIEW2_TEST = np.zeros((VIEW2_TEST.shape[0], VIEW2_TEST.shape[1]*NUM))

    index_shift = 0
    index_shift2 = 0
    index_shift3 = 0
    for index1 in range(LEN):
        if index1<CUT1:
            for index2 in range(LENGTH[index1][0]):
                for index3 in range(NUM):
                    tindex = index2 + index3 - HALF
                    if (tindex>=0) and (tindex<LENGTH[index1][0]):
                        VVIEW1_TRAIN[index_shift+index2][index3*VIEW1_TRAIN.shape[1]:(index3+1)*VIEW1_TRAIN.shape[1]] = VIEW1_TRAIN[index_shift+tindex][0:VIEW1_TRAIN.shape[1]]
                        VVIEW2_TRAIN[index_shift+index2][index3*VIEW2_TRAIN.shape[1]:(index3+1)*VIEW2_TRAIN.shape[1]] = VIEW2_TRAIN[index_shift+tindex][0:VIEW2_TRAIN.shape[1]]
            index_shift += LENGTH[index1][0]
        #elif index1<CUT2:
        #    for index2 in range(LENGTH[index1][0]):
        #        for index3 in range(NUM):
        #            tindex = index2 + index3 - HALF
        #            if (tindex>=0) and (tindex<LENGTH[index1][0]):
        #                VVIEW1_VALIDATE[index_shift2+index2][index3*VIEW1_VALIDATE.shape[1]:(index3+1)*VIEW1_VALIDATE.shape[1]] = VIEW1_VALIDATE[index_shift2+tindex][0:VIEW1_VALIDATE.shape[1]]
        #                VVIEW2_VALIDATE[index_shift2+index2][index3*VIEW2_VALIDATE.shape[1]:(index3+1)*VIEW2_VALIDATE.shape[1]] = VIEW2_VALIDATE[index_shift2+tindex][0:VIEW2_VALIDATE.shape[1]]
        #    index_shift2 += LENGTH[index1][0]
        #else:
        #    for index2 in range(LENGTH[index1][0]):
        #        for index3 in range(NUM):
        #            tindex = index2 + index3 - HALF
        #            if (tindex>=0) and (tindex<LENGTH[index1][0]):
        #                VVIEW1_TEST[index_shift3+index2][index3*VIEW1_TEST.shape[1]:(index3+1)*VIEW1_TEST.shape[1]] = VIEW1_TEST[index_shift3+tindex][0:VIEW1_TEST.shape[1]]
        #                VVIEW2_TEST[index_shift3+index2][index3*VIEW2_TEST.shape[1]:(index3+1)*VIEW2_TEST.shape[1]] = VIEW2_TEST[index_shift3+tindex][0:VIEW2_TEST.shape[1]]
        #    index_shift3 += LENGTH[index1][0]

    train=DataSet(VVIEW1_TRAIN,VVIEW2_TRAIN,LABEL_TRAIN,train_z_mean,train_z_var,train_h1_mean,train_h1_var,train_h2_mean,train_h2_var)    
    #tune=DataSet(VVIEW1_VALIDATE,VVIEW2_VALIDATE,LABEL_VALIDATE,tune_z_mean,tune_z_var,tune_h1_mean,tune_h1_var,tune_h2_mean,tune_h2_var)
    #test=DataSet(VVIEW1_TEST,VVIEW2_TEST,LABEL_TEST,test_z_mean,test_z_var,test_h1_mean,test_h1_var,test_h2_mean,test_h2_var)
    tune=None
    test=None
    return train, tune, test



trainData,tuneData,testData=read_xrmb()