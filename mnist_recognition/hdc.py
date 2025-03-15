class Hdc(object):
    def __init__(self, N, xp):
        self.N = N
        self.xp = xp
        self.bundle = self.bundle_noties

    def __call__(self, n=1):
        size = self.N if n==1 else (n, self.N)
        return self.xp.random.default_rng().choice(self.xp.array([-1, +1], dtype='b'), size=size)

    def normalize(self, hdv):
        if type(hdv) is list:
            hdv = self.xp.array(hdv)
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

    def absdist(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return self.xp.count_nonzero(hdv1 != hdv2)

    def reldist(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return self.xp.count_nonzero(hdv1 != hdv2) / self.N

    def bundle_ties(self, hdv1, *hdvs):
        if type(hdv1) is list: # bundle([x1, x2])
            assert not hdvs
            hdvs = hdv1
        else:
            match hdv1.shape:
                case (_, self.N): # bundle(matrix_of_hdvs)
                    hdvs = hdv1
                case (self.N,): # bundle(x1, x2, x3)
                    assert len(hdvs) > 0
                    t = hdvs
                    hdvs = [hdv1]
                    hdvs.extend(t)
                case _:
                    assert False, hdv1.shape
    
        sum = self.xp.sum(hdvs, axis=0)
        
        if len(hdvs) % 2 == 0:
            tie_breaker = self()
            sum = self.xp.sum([sum, tie_breaker], axis=0)
            
        return self.xp.sign(sum)

    def bundle_noties(self, hdv1, *hdvs):
        if type(hdv1) is list: # bundle([x1, x2])
            assert not hdvs
            hdvs = hdv1
        else:
            match hdv1.shape:
                case (_, self.N): # bundle(matrix_of_hdvs)
                    hdvs = hdv1
                case (self.N,): # bundle(x1, x2, x3)
                    assert len(hdvs) > 0
                    t = hdvs
                    hdvs = [hdv1]
                    hdvs.extend(t)
                case _:
                    assert False, hdv1.shape
    
        sum = self.xp.sum(hdvs, axis=0)
        return self.xp.sign(sum)

    def debundle(self, hdv_bundle, hdv):
        assert hdv_bundle.shape == (self.N,)
        assert hdv.shape == (self.N,)
        complement = hdv * (-1)
        return self.bundle(hdv_bundle, complement)

    def bind(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return self.xp.prod([hdv1, hdv2], axis=0)
        
    def shift(self, hdv, k=1):
        assert hdv.shape == (self.N,)
        return self.xp.roll(hdv, k)

    def sim(self, hdv1, hdv2):
        assert hdv1.shape == (self.N,)
        assert hdv2.shape == (self.N,)
        return hdv1.astype(int) @ hdv2.astype(int) / (self.xp.linalg.norm(hdv1) *  self.xp.linalg.norm(hdv2)) # .astype(int) is a MUST, otherwise Geisenbugs with overflow occur

class HdvArray(object):
    def __init__(self, N, xp):
        self.xp = xp
        self.N = N
        self.array = xp.zeros((10, N))
        self.free_slots = set(range(self.array.shape[0]))

    def len(self):
        return self.array.shape[0] - len(self.free_slots)

    def lease(self):
        if self.free_slots:
            return self.free_slots.pop()

        current_array_size = self.array.shape[0]
        new_array_size = current_array_size * 2
        new_array = self.xp.zeros((new_array_size, self.N))
        new_array[:current_array_size] = self.array
        self.array = new_array
        self.free_slots.update(range(current_array_size, self.array.shape[0]))
        return self.free_slots.pop()

    def release(self, index):
        assert not index in self.free_slots
        self.free_slots.add(index)
        self.array[index] = self.xp.zeros(self.N)
        
        
