import numpy as np

class Hdc(object):
    def __init__(self, N, xp, dtype='b'):
        self.N = N
        self.xp = xp
        self.dtype = dtype
        self.source = np.array([-1, +1], dtype=self.dtype)

        if self.xp.__name__ == 'cupy':
            self.wrap_list = lambda l: self.xp.array(l)
            self.wrap_array = lambda a: self.xp.asarray(a)
        else:
            self.wrap_list = lambda l: l
            self.wrap_array = lambda a: a

    def __call__(self, n=None):
        size = self.N if n is None else (n, self.N)
        hdv = np.random.default_rng().choice(self.source, size=size) # RNG.choice on host (CPU) side is much faster than on GPU
        return self.wrap_array(hdv)

    def zero(self, n=1):
        size = self.N if n==1 else (n, self.N)
        return self.xp.zeros(size, dtype=self.dtype) 

    def normalize(self, hdv):
        if type(hdv) is list:
            hdv = self.xp.array(hdv) # no wrap_list since at the end we do batch division with broadcast
        else:
            match hdv.shape:
                case (N,):
                    l = self.xp.linalg.norm(hdv)
                    return hdv / l if l > 0 else hdv
                case (_, N):
                    pass
                case _:
                    assert False, hdv.shape
    
        l = self.xp.linalg.norm(hdv, axis=1)
        return (hdv.T / l).T

    def complement(self, hdv):
        assert hdv.shape == (self.N,)
        return hdv * (-1)

    # Suitable with hdist while sim will produce lower numbers due to random -1/+1 deployed
    def bundle_ties(self, hdv1, *hdvs):
        if type(hdv1) is list: # bundle([x1, x2])
            assert not hdvs # hdvs must be empty (not None!)
            hdvs = self.wrap_list(hdv1)
        else:
            match hdv1.shape:
                case (_, self.N): # bundle(matrix_of_hdvs)
                    assert not hdvs, f'First argument is already a matrix of HDVs, there must be no second argument (got {type(hdvs)})'
                    hdvs = hdv1
                case (self.N,): # bundle(x1, x2, x3)
                    assert len(hdvs) > 0
                    t = hdvs
                    hdvs = [hdv1]
                    hdvs.extend(t)
                    hdvs = self.wrap_list(hdvs)
                case _:
                    assert False, hdv1.shape

        sum = self.xp.sum(hdvs, axis=0)
        
        if len(hdvs) % 2 == 0:
            tie_breaker = self()
            sum = self.xp.sum(self.wrap_list([sum, tie_breaker]), axis=0)
            
        return self.xp.sign(sum).astype(self.dtype)

    # Suitable with sim while hdist will flicker due to even / odd number of hdvs in summation
    def bundle_noties(self, hdv1, *hdvs):
        if type(hdv1) is list: # bundle([x1, x2])
            assert not hdvs # hdvs must be empty (not None!)
            hdvs = self.wrap_list(hdv1)
        else:
            match hdv1.shape:
                case (_, self.N): # bundle(matrix_of_hdvs)
                    assert not hdvs, f'First argument is already a matrix of HDVs, there must be no second argument (got {type(hdvs)})'
                    hdvs = hdv1
                case (self.N,): # bundle(x1, x2, x3)
                    assert len(hdvs) > 0
                    t = hdvs
                    hdvs = [hdv1]
                    hdvs.extend(t)
                    hdvs = self.wrap_list(hdvs)
                case _:
                    assert False, hdv1.shape

        sum = self.xp.sum(hdvs, axis=0)
        return self.xp.sign(sum).astype(self.dtype)

    def debundle(self, hdv_bundle, hdv):
        assert hdv_bundle.shape == (self.N,)
        assert hdv.shape == (self.N,)
        complement = hdv * (-1)
        return self.bundle(hdv_bundle, complement)

    def bind(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return self.xp.prod(self.xp.array([hdv1, hdv2]), axis=0).astype(self.dtype)
        
    def shift(self, hdv, k=1):
        assert hdv.shape == (self.N,)
        return self.xp.roll(hdv, k).astype(self.dtype)

    def to_binary(self, bipolar_hdv):
        tie_breaker = self()
        t = self.xp.where(bipolar_hdv == 0, tie_breaker, bipolar_hdv) # [-1, 0, +1, 0, ...] -> [-1, rnd(-1,+1), 1, rnd(-1,+1), ...]
        return (t > 0).astype(self.dtype) # [-1, +1, +1, -1, ...] -> [0, 1, 1, 0, ...]

    def to_bipolar(self, binary_hdv):
        return self.xp.where(binary_hdv == 0, -1, +1)

    # Hamming distance
    def hdist(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return self.xp.count_nonzero(hdv1 != hdv2)
    
    # Cosine similarity
    def sim(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return hdv1.astype('f') @ hdv2.astype('f') / (self.xp.linalg.norm(hdv1) * self.xp.linalg.norm(hdv2)) # .astype('f') is a MUST, otherwise Geisenbugs with overflow may occur
