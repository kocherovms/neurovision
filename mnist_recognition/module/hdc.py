from heapq import heapify, heappush, heappop

class Hdc(object):
    COS_SIM_THRESHOLD = 0.055
    HDIST_THRESHOLD = 4700
    
    def __init__(self, N, xp):
        self.N = N
        self.xp = xp
        self.source = self.xp.array([-1, +1], dtype='b')
        self.bundle = self.bundle_noties

        if self.xp.__name__ == 'cupy':
            self.wrap_list = lambda l: self.xp.array(l)
        else:
            self.wrap_list = lambda l: l

    def __call__(self, n=None):
        size = self.N if n is None else (n, self.N)
        return self.xp.random.choice(self.source, size=size) # no cp.random.default_rng().choice in cupy :(

    def zero(self, n=1):
        size = self.N if n==1 else (n, self.N)
        return self.xp.zeros(size, dtype='b') 

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

    # Hamming distance
    def hdist(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return self.xp.count_nonzero(hdv1 != hdv2)

    # Suitable with hdist, sim will produce lower numbers due to random -1/+1 deployed
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
            
        return self.xp.sign(sum).astype('b')

    # Suitable with sim. hdist will flicker due to even / odd number of hdvs in summation
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
        return self.xp.sign(sum).astype('b')

    def debundle(self, hdv_bundle, hdv):
        assert hdv_bundle.shape == (self.N,)
        assert hdv.shape == (self.N,)
        complement = hdv * (-1)
        return self.bundle(hdv_bundle, complement)

    def bind(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return self.xp.prod(self.xp.array([hdv1, hdv2]), axis=0).astype('b')
        
    def shift(self, hdv, k=1):
        assert hdv.shape == (self.N,)
        return self.xp.roll(hdv, k).astype('b')

    # Cosine similarity
    def sim(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return hdv1.astype(int) @ hdv2.astype(int) / (self.xp.linalg.norm(hdv1) *  self.xp.linalg.norm(hdv2)) # .astype(int) is a MUST, otherwise Geisenbugs with overflow may occur

class HdvArray(object):
    def __init__(self, N, xp, initial_length=10, dtype=None, grow_policy=None):
        self.xp = xp
        self.N = N
        self.array = xp.zeros((initial_length, N), dtype=dtype)
        self.free_indices = list(range(self.array.shape[0]))
        heapify(self.free_indices)
        self.leased_indices = set()
        self.max_leased_index = -1
        self.grow_policy = (lambda current_array_size: current_array_size * 2) if grow_policy is None else grow_policy

    @property
    def len(self):
        return len(self.leased_indices)

    @property 
    def active_len(self):
        return self.max_leased_index + 1

    def lease(self):
        if self.free_indices:
            index = heappop(self.free_indices) # lowest possible index is returned, so the array space is reused effectively
            assert not index in self.leased_indices 
            self.leased_indices.add(index)
            self.max_leased_index = max(self.max_leased_index, index)
            return index

        current_array_size = self.array.shape[0]
        new_array_size = self.grow_policy(current_array_size)
        assert new_array_size > current_array_size
        
        # new_array = self.xp.zeros((new_array_size, self.N), self.array.dtype)
        # new_array[:current_array_size] = self.array
        # self.array = new_array

        self.array = self.xp.resize(self.array, (new_array_size, self.N))
        self.array[current_array_size:] = 0

        for free_index in range(current_array_size, self.array.shape[0]):
            heappush(self.free_indices, free_index)

        index = heappop(self.free_indices) # lowest possible index is returned, so the array space is reused effectively
        assert not index in self.leased_indices 
        self.leased_indices.add(index)
        self.max_leased_index = max(self.max_leased_index, index)
        return index

    def release(self, index):
        assert index in self.leased_indices
        heappush(self.free_indices, index)
        self.leased_indices.discard(index)
        
        if index == self.max_leased_index:
            # max on set of say 50k takes xxx microseconds. So rescan only when max_leased_index was popped!
            self.max_leased_index = max(self.leased_indices) if self.leased_indices else -1
            
        self.array[index] = 0

    def clear(self):
        self.free_indices = list(range(self.array.shape[0]))
        heapify(self.free_indices)
        self.leased_indices = set()
        self.max_leased_index = -1
        self.array[:] = 0

    @property
    def array_active(self):
        return self.array[:self.active_len]
