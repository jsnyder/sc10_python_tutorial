import pyopencl as cl
import pyopencl.clrandom
import numpy as np

nsamples = int(12e6)

# set up context and queue
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# create array of random values in OpenCL
xy = pyopencl.clrandom.rand(ctx,queue,(nsamples,2),np.float32)

# square values in OpenCL
xy = xy**2

# 'get' method on xy is used to get array from OpenCL into ndarray
print 4.0*np.sum(np.sum(xy.get(),1)<1)/nsamples
