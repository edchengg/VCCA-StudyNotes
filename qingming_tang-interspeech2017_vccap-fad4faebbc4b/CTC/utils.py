import numpy as np

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []

    for i, seq in enumerate(sequences):
        indices.extend(zip([i] * len(seq), xrange(len(seq))))
        values.extend(seq)
    
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape
