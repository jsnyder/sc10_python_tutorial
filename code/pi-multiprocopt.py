import multiprocessing as mp
import numpy as np

processes = mp.cpu_count()
nsamples = int(12e6/processes)

def calcInsideNumPy(rank):
    np.random.seed(rank)

    # "vectorized" sample gen, col 0 = x, col 1 = y
    xy = np.random.random((nsamples,2))
    return 4.0*np.sum(np.sum(xy**2,1)<1)/nsamples

if __name__ == '__main__':
    pool = mp.Pool(processes)
    result = pool.map(calcInsideNumPy, range(processes))
    print np.mean(result)
