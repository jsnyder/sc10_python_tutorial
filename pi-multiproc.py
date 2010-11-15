import multiprocessing as mp
import numpy as np
import random

processes = mp.cpu_count()
nsamples = int(12e6/processes)

def calcInside(rank):
    inside = 0
    random.seed(rank)
    for i in range(nsamples):
        x = random.random();
        y = random.random();
        if (x*x)+(y*y)<1:
            inside += 1
    return (4.0*inside)/nsamples
           
if __name__ == '__main__':
    pool = mp.Pool(processes)
    result = pool.map(calcInside, range(processes))
    print np.mean(result)
