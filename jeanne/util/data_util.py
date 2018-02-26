import numpy as np

def pad_idx(indices, batchsize) :
    if batchsize == 0 :
        return indices
    assert len(indices) >= batchsize, "batchsize must be less or equal than number of data sample"
    if len(indices) % batchsize == 0 :
        return indices
    num_remaining = batchsize - (len(indices) % batchsize)
    return indices+indices[0:num_remaining]
    pass

def one_hot(x, m=None) :
    if m is None :
        m = np.max(x)+1
    mat = np.zeros((len(x), m))
    mat[np.arange(len(x)), x] = 1.0
    return mat

def stat_array(x, axis=0) :
    return {'min':x.min(axis=axis), 'max':x.max(axis=axis), 'mean':x.mean(axis=axis), 'std':x.std(axis=axis)}

class IncrementalMax :
    def __init__(self, axis=0) :
        self.axis = axis
        self.stat_max = None
        pass
    
    def update(self, x) :
        new_max = np.max(x, axis=self.axis)
        if self.stat_max is not None :
            new_max = np.max([self.stat_max, new_max], axis=0)
        self.stat_max = new_max
        pass

class IncrementalMin :
    def __init__(self, axis=0) :
        self.axis = axis
        self.stat_min = None
        pass
    
    def update(self, x) :
        new_min = np.min(x, axis=self.axis)
        if self.stat_min is not None :
            new_min = np.min([self.stat_min, new_min], axis=0)
        self.stat_min = new_min
        pass


# DATA ITERATOR #
def iter_minibatches(indices, batchsize, shuffle=True, use_padding=True, excludes=None):
    """
    Args:
        datasize : total number of data or list of indices
        batchsize : mini-batchsize
        shuffle :
        use_padding : pad the dataset if dataset can't divided by batchsize equally

    Return :
        list of index for current epoch (randomized or not depends on shuffle)
    """
    if isinstance(indices, list) :
        indices = indices
    elif isinstance(indices, int) :
        indices = list(range(indices))
    if excludes is not None :
        indices = [x for x in indices if x not in excludes]
    if shuffle:
        np.random.shuffle(indices)

    if use_padding :
        indices = pad_idx(indices, batchsize)

    for ii in range(0, len(indices), batchsize):
        yield indices[ii:ii + batchsize]
    pass

def iter_minibucket(indices, batchsize, shuffle=True, excludes=None) :
    """
    Iterate bucket of index for efficient different sequence training
    Notes : returned bucket must be retrieved on sorted index
    Example :

    x = [datasize x seqlen x ndim]
    sidx = np.argsort(map(len, x))
    for rr in iter_minibucket(datasize, batchsize) :
        curr_x = [x[sidx[ii]] for ii in rr] # get sorted index and retrieve x
    """
    if isinstance(indices, list) :
        indices = indices
    elif isinstance(indices, int) :
        indices = list(range(indices))
    if excludes is not None :
        indices = [x for x in indices if x not in excludes]
    datasize = len(indices)
    indices = [indices[ii:ii+batchsize] for ii in range(0, datasize, batchsize)]
    if shuffle :
        np.random.shuffle(indices)
    for ii in range(0, len(indices)) :
        yield indices[ii]
    pass 

def context_window(seq, size=5) :
    """
    Create list of overlap context window given sequence
    Primary used in signal processing

    Args :
        seq : list of frames [time x n-dim vector]
        size : range context window to the left and right (total frame = 1 + size * 2)
    """
    total = 1 + size*2
    ndim = seq.shape[1]
    result = np.zeros((seq.shape[0], seq.shape[1]*(total)))
    mid = size
    for ii in range(len(seq)) :
        left_seq = max(ii-size, 0)
        right_seq = min(ii+size, len(seq)-1)
        left_con = left_seq-(ii-size)
        right_con = total-(ii+size-right_seq)
        # print left_seq, right_seq
        # print left_con, right_con
        result[ii][left_con*ndim:right_con*ndim] = seq[left_seq:right_seq+1].flatten()
    return result
    pass

