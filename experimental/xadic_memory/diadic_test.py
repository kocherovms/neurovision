# Test for DiadicMemory
"""
The xcount variable below sets how many records would be stored & queried within memory
"""
import argparse

from time import time
import numpy as np
from mem_sdrsdm import DiadicMemory as MemDiadicMemory
from sdrsdm import DiadicMemory
from sdr_util import random_sdrs


def test_diadic(mem, xcount):
    print(f"Testing DiadicMemory for {xcount} entries, sdr size is {mem.N} with {mem.P} of ON bits\n")
    # The number of records to write and query

    t = time()
    X = random_sdrs(xcount+1, sdr_size = mem.N, on_bits = mem.P)
    t = time() - t
    print(f"{xcount+1} random SDRs generated in {int(t*1000)}ms\n")

    # Store chain of sdrs: X[0] -> X[1], X[1] -> X[2], ...
    t = time()
    for i,x in enumerate(X):
        if i == xcount:
            break
        y = X[i+1]
        mem.store(x,y)   ######################################################## Store operation
    t = time() - t
    print(f"{xcount} writes in {int(t*1000)} ms\n")

    # print("Testing queries")

    size_errors = {}
    found = np.zeros((xcount,mem.P),dtype = np.uint16)
    t = time()

    for i in range(xcount): 
        qresult = mem.query(X[i]) ########################################## Query operations
        if qresult.size == mem.P:
            found[i] = qresult
        else:
            found[i] = X[i+1] #  Initial mapping value
            size_errors[i] = qresult

    t = time() - t

    print(f"{xcount} queries done in {int(t*1000)}ms\n")

    # print("Comparing results with expectations")
    
    if len(size_errors): 
        print(f"{len(size_errors)} size errors, check size_errors dictionary")
        
    diff = (X[1:] != found)
    diffs_count = (diff.sum(axis=1) > 0).sum()

    if diffs_count > 0:
        print(f"{diffs_count} differences check diff array")

    if len(size_errors) <=0 and diffs_count <= 0:
        print('Tests OK')

parser = argparse.ArgumentParser(
                    prog='diadic tester',
                    description='test the diadic memory')

parser.add_argument('-s', '--size', type=int, default=1000) # SDR_SIZE = 1000  # Size of SDRs in bits
parser.add_argument('-b', '--bits', type=int, default=10) # SDR_BITS =   10  # Default number of ON bits (aka solidity)
parser.add_argument('-c', '--count', type=int, default=45000)
parser.add_argument('-m', '--mem', action='store_true')
args = parser.parse_args()

if args.mem:
    mem = MemDiadicMemory(args.size, args.bits) ######################################## Initialize the memory
else:
    mem = DiadicMemory(args.size, args.bits)

test_diadic(mem, args.count)