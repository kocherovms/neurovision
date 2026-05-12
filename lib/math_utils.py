from collections import deque
import numpy as np

import lang_utils as lu

def softmax(x):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

def conflate(pdfs):    
    n = np.prod(pdfs, axis=0)
    d = n.sum()

    if np.isclose(d, 0):
        return np.zeros(len(pdfs))
        
    return n / d

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret[:n-1] = np.array(a[:n-1]) * n
    return ret / n

def get_angle_diff(a_from, a_to):
    andiff = a_to - a_from
    return (andiff + 180) % 360 - 180

# Dr. Shane Ross explained this beautifuly in https://www.youtube.com/watch?v=HCd-leV8OkU
# Also improvement is made to support batched input
class RecursiveAverageFilter:
    def __init__(self):
        self.n = 0
        self.v = 0

    def __call__(self, x, batch_size=1):
        n_new = self.n + batch_size
        self.v = (self.n * self.v + batch_size * x) / n_new
        self.n = n_new
        return self.v

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return f'RecursiveAverageFilter(v={self.v}, n={self.n})'

class RecursiveMovingAverageFilter:
    def __init__(self, max_n):
        self.v = 0
        self.max_n = max_n
        self.elems = deque(maxlen=max_n)

    def __call__(self, x):
        if not self.elems:
            self.v = x
        else:
            first_elem = lu.when(len(self.elems) == self.max_n, lambda: self.elems[0], None)
            # Logic is based on idea: AVG(elems) * LEN(elems) = SUM(elems)
            window_sum = self.v * len(self.elems) - lu.coalesce(first_elem, 0) + x
            self.v = window_sum / min(len(self.elems) + 1, self.max_n)
            
        self.elems.append(x)
        return self.v

    def __str__(self):
        return str(self.v)

    def __repr__(self):
        return f'RecursiveMovingAverageFilter(v={self.v}, n={len(self.elems)}, max_n={self.max_n})'
    