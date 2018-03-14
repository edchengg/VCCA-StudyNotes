import scipy.io as sio
import tensorflow as tf
import numpy as np

class DataSet(object):
    
    def __init__(self, images, labels, lengths, dtype=tf.float32):
        dtype = tf.as_dtype(dtype).base_dtype
        if dtype not in (tf.uint8, tf.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)
        # _num_examples is the total frame
        self._num_examples = images.shape[0]
        # _num_seqs is the total sequences
        self._num_seqs = lengths.shape[0]
        self._images = images
        self._labels = labels
        self._lengths = lengths
        # which sequence?
        self._index_in_epoch = 0
        # which frame?
        self._index_total_in_epoch = 0
        # number of frames in each batch
        self._this_batch_size = 0
        self._parray = np.arange(self._num_seqs)
        self._agg = np.zeros(self._num_seqs)
        self._agg[0] = 0
        for index in range(self._num_seqs-1):
            self._agg[index+1] = self._agg[index]+self._lengths[index]

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels
    
    @property
    def lengths(self):
        return self._lengths

    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def num_seqs(self):
        return self._num_seqs

    def rshuffle(self):
        np.random.shuffle(self._parray)

    def next_batch(self, batch_size, start_pos):
        self._this_batch_size = 0
        v = self._images[-1:-1]
        l = self._labels[-1:-1]
        maxsize = -1
        tmplen = np.zeros((batch_size),dtype=np.int)
        self._index_in_epoch = start_pos
        # Find max len
        for index in range(batch_size):
            if self._lengths[self._parray[(self._index_in_epoch+index)%self._num_seqs]][0]>maxsize:
                maxsize = self._lengths[self._parray[(self._index_in_epoch+index)%self._num_seqs]][0]
            tmplen[index] = self._lengths[self._parray[(self._index_in_epoch+index)%self._num_seqs]][0]
        # Construct batch
        for index in range(batch_size):
            self._index_in_epoch %= self._num_seqs
            start = self._agg[self._parray[self._index_in_epoch]]
            end = start + self._lengths[self._parray[self._index_in_epoch]][0]
            self._this_batch_size += self._lengths[self._parray[self._index_in_epoch]][0]
            v = np.concatenate((v,self._images[start:end]),0)
            l = np.concatenate((l,self._labels[start:end]),0)
            self._index_in_epoch += 1
        vv = np.zeros((batch_size,maxsize,self._images.shape[1]))
        ll = np.zeros((batch_size,maxsize,1),dtype=int)
        # Reformat
        index_sum = 0
        for index in range(batch_size):
            for iindex in range(tmplen[index]):
                vv[index][iindex] = v[index_sum+iindex]
                ll[index][iindex][0] = l[index_sum+iindex][0]
            index_sum += tmplen[index]
        ll = np.reshape(ll, (np.shape(ll)[0], np.shape(ll)[1]))
        maxsize = -1
        for index in range(batch_size):
            cur_value=-1
            tmpsize = 0
            for iindex in range(tmplen[index]):
                if (cur_value != ll[index][iindex]):
                    tmpsize += 1
                    cur_value = ll[index][iindex]
            if tmpsize>maxsize:
                maxsize=tmpsize
        lll = np.zeros((batch_size,maxsize),dtype=int)
        for index in range(batch_size):
            cur_value=-1
            tmpsize=0
            for iindex in range(tmplen[index]):
                if (cur_value != ll[index][iindex]):
                    cur_value = ll[index][iindex]
                    lll[index][tmpsize] = cur_value
                    tmpsize += 1
        return vv, lll, tmplen, self._this_batch_size

def read_xrmb(fold=0):
    data=sio.loadmat('./XRMB_SEQ.mat')
    LENGTH = data['LENGTHS']
    LABEL = data['LABELS']
    VIEW = data['MFCCS']
    CUT1 = 1500
    CUT2 = 1736
    LEN = 2357
    TOTAL_FRAME = 2430668

    LABEL_TEST = LABEL[-1:-1]
    LABEL_TRAIN = LABEL[-1:-1]
    LABEL_DEV = LABEL[-1:-1]
    LENGTH_TEST = LENGTH[-1:-1]
    LENGTH_TRAIN = LENGTH[-1:-1]
    LENGTH_DEV = LENGTH[-1:-1]
    VIEW_TEST = VIEW[-1:-1]
    VIEW_TRAIN = VIEW[-1:-1]
    VIEW_DEV = VIEW[-1:-1]

    num = 0
    numlable = 0
    for index in range(CUT2):
        numlable += LENGTH[index][0]
    print np.shape(VIEW)
    print numlable
    print TOTAL_FRAME
    FOLDNUM = 104
    for lenindex in range(CUT2,LEN):
        if ((lenindex < CUT2+FOLDNUM*(fold+1)) and (lenindex >= CUT2+FOLDNUM*fold)):
            LENGTH_TEST = np.concatenate((LENGTH_TEST,LENGTH[lenindex:lenindex+1]),0)
            LABEL_TEST = np.concatenate((LABEL_TEST,LABEL[numlable:numlable+LENGTH[lenindex][0]]),0)
            VIEW_TEST = np.concatenate((VIEW_TEST,VIEW[numlable:numlable+LENGTH[lenindex][0]]),0)
        elif ((lenindex >= CUT2+FOLDNUM*((fold+1)%6)) and (lenindex < CUT2+FOLDNUM*((fold+1)%6+1))):
            LENGTH_DEV = np.concatenate((LENGTH_DEV,LENGTH[lenindex:lenindex+1]),0)
            LABEL_DEV = np.concatenate((LABEL_DEV,LABEL[numlable:numlable+LENGTH[lenindex][0]]),0)
            VIEW_DEV = np.concatenate((VIEW_DEV,VIEW[numlable:numlable+LENGTH[lenindex][0]]),0)
        else:
            LENGTH_TRAIN = np.concatenate((LENGTH_TRAIN,LENGTH[lenindex:lenindex+1]),0)
            LABEL_TRAIN = np.concatenate((LABEL_TRAIN,LABEL[numlable:numlable+LENGTH[lenindex][0]]),0)
            VIEW_TRAIN = np.concatenate((VIEW_TRAIN,VIEW[numlable:numlable+LENGTH[lenindex][0]]),0)
        num += LENGTH[lenindex][0]
        numlable += LENGTH[lenindex][0]

    test=DataSet(VIEW_TEST, LABEL_TEST, LENGTH_TEST)
    dev=DataSet(VIEW_DEV, LABEL_DEV, LENGTH_DEV)
    train=DataSet(VIEW_TRAIN, LABEL_TRAIN, LENGTH_TRAIN)

    return train,dev,test

