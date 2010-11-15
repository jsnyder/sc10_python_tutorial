from mpi4py import MPI
import numpy as np
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
mpisize = comm.Get_size()
nsamples = int(12e6/mpisize)

inside = 0
random.seed(rank)
for i in range(nsamples):
    x = random.random()
    y = random.random()
    if (x*x)+(y*y)<1:
      inside += 1

mypi = (4.0 * inside)/nsamples  
pi = comm.reduce(mypi, op=MPI.SUM, root=0)
  
if rank==0:
    print (1.0 / mpisize)*pi
