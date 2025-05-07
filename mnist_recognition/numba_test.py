from numba import jit
import numpy as np
import time

@jit(nopython=True)
def go_fast(l):
    s = 0
    
    for e in l:
        s += e

    return s

def go_ordinary(l):
    s = 0
    
    for e in l:
        s += e

    return s

@jit(nopython=True)
def go2_fast(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

def go2_ordinary(a): # Function is compiled and runs in machine code
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace

l = list(range(1000000))

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.perf_counter()
go_fast(l)
end = time.perf_counter()
print("go_fast. Elapsed (with compilation) = {}s".format((end - start)))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.perf_counter()
go_fast(l)
end = time.perf_counter()
print("go_fast. Elapsed (after compilation) = {}s".format((end - start)))

start = time.perf_counter()
go_ordinary(l)
end = time.perf_counter()
print("go_ordinary. Elapsed (after compilation) = {}s".format((end - start)))




x = np.arange(100).reshape(10, 10)

# DO NOT REPORT THIS... COMPILATION TIME IS INCLUDED IN THE EXECUTION TIME!
start = time.perf_counter()
go2_fast(x)
end = time.perf_counter()
print("go2_fast. Elapsed (with compilation) = {}s".format((end - start)))

# NOW THE FUNCTION IS COMPILED, RE-TIME IT EXECUTING FROM CACHE
start = time.perf_counter()
go2_fast(x)
end = time.perf_counter()
print("go2_fast. Elapsed (after compilation) = {}s".format((end - start)))

start = time.perf_counter()
go2_ordinary(x)
end = time.perf_counter()
print("go2_ordinary. Elapsed (after compilation) = {}s".format((end - start)))