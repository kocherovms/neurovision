"""
This code is an implementation of the Triadic Memory and Dyadic Memory algorithms

Copyright (c) 2021-2022 Peter Overmann
Copyright (c) 2022 Cezar Totth
Copyright (c) 2023 Clément Michaud

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import numba
from collections import defaultdict

def xaddr(x):
    addr = []
    for i in range(1,len(x)):
        for j in range(i):
            yield x[i] * (x[i] - 1) // 2 + x[j]

@numba.jit(nopython=True)
def sums2sdr(sums, P):
    # this does what binarize() does in C
    ssums = sums.copy()
    ssums.sort()
    threshval = ssums[-P]
    if threshval == 0:
        return np.where(sums)[0]  # All non zero values
    else:
        return np.where(sums >= threshval)[0] # 


class DiadicMemory_Orig:
    """
    this is a convenient object front end for SDM functions
    """
    def __init__(self, N, P):
        """
        N is SDR vector size, e.g. 1000
        P is the count of solid bits e.g. 10
        """
        self.mem = defaultdict(lambda: defaultdict(lambda: 0))
        self.N = N
        self.P = P

    def store(self, x, y):
        for addr in xaddr(x):
            for j in y:
                self.mem[addr][j] = 1

    def query(self, x):
        sums = np.zeros(self.N, dtype=np.uint32)
        for addr in xaddr(x):
            for k, v in self.mem[addr].items():
                sums[k] += v
        return sums2sdr(sums, self.P)

class DiadicMemory_Counters:
    """
    this is a convenient object front end for SDM functions
    """
    def __init__(self, N, P):
        """
        N is SDR vector size, e.g. 1000
        P is the count of solid bits e.g. 10
        """
        self.mem = defaultdict(lambda: np.zeros(N, dtype=int))
        self.N = N
        self.P = P

    def store(self, x, y):
        y_hat = np.full(self.N, 0, dtype=int)
        y_hat[y] = 1
        
        for addr in xaddr(x):
            counters = self.mem[addr]
            counters += y_hat

    def query(self, x):
        sums = np.zeros(self.N, dtype=int)
        
        for addr in xaddr(x):
            counters = self.mem[addr]
            sums += counters

        return sums2sdr(sums, self.P)

class DiadicMemory_SdmCounters:
    """
    this is a convenient object front end for SDM functions
    """
    def __init__(self, N, P):
        """
        N is SDR vector size, e.g. 1000
        P is the count of solid bits e.g. 10
        """
        self.mem = defaultdict(lambda: np.zeros(N, dtype=int))
        self.N = N
        self.P = P

    def store(self, x, y):
        y_hat = np.full(self.N, -1, dtype=int) # SDM Counter!
        y_hat[y] = +1
        
        for addr in xaddr(x):
            counters = self.mem[addr]
            counters += y_hat

    def query(self, x):
        sums = np.zeros(self.N, dtype=int)
        
        for addr in xaddr(x):
            counters = self.mem[addr]
            sums += counters

        return sums2sdr(sums, self.P)

class DiadicMemory_SdmCountersAndRetrieval:
    """
    this is a convenient object front end for SDM functions
    """
    def __init__(self, N, P):
        """
        N is SDR vector size, e.g. 1000
        P is the count of solid bits e.g. 10
        """
        self.mem = defaultdict(lambda: np.zeros(N, dtype=int))
        self.N = N
        self.P = P

    def store(self, x, y):
        y_hat = np.full(self.N, -1, dtype=int) # SDM Counter!
        y_hat[y] = +1
        
        for addr in xaddr(x):
            counters = self.mem[addr]
            counters += y_hat

    def query(self, x):
        rv = np.random.randint(2, size=self.N, dtype=int)
        sums = np.zeros(self.N, dtype=int)
        
        for addr in xaddr(x):
            counters = self.mem[addr]
            sums += counters

        rv[sums > 0] = 1
        rv[sums < 0] = 0
        return np.where(rv > 0)[0]