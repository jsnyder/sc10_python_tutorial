import pyopencl as cl
import numpy as np
import sys, time
import matplotlib
import matplotlib.pyplot as plt

class Grid:
    """A simple grid class that stores the details and solution of the
    computational grid."""
    def __init__(self, nx=10, ny=10, xmin=0.0, xmax=1.0,
                 ymin=0.0, ymax=1.0):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.dx = float(xmax-xmin)/(nx-1)
        self.dy = float(ymax-ymin)/(ny-1)
        self.u = np.zeros((nx, ny)).astype(np.float32)
        # used to compute the change in solution in some of the methods.
        self.old_u = self.u.copy()

    def setBCFunc(self, func):
        """Sets the BC given a function of two variables."""
        xmin, ymin = self.xmin, self.ymin
        xmax, ymax = self.xmax, self.ymax
        x = np.arange(xmin, xmax + self.dx*0.5, self.dx)
        y = np.arange(ymin, ymax + self.dy*0.5, self.dy)
        self.u[0 ,:] = func(xmin,y)
        self.u[-1,:] = func(xmax,y)
        self.u[:, 0] = func(x,ymin)
        self.u[:,-1] = func(x,ymax)

    def computeError(self):
        """Computes absolute error using an L2 norm for the solution.
        This requires that self.u and self.old_u must be appropriately
        setup."""
        v = (self.u[1:-1, 1:-1] - self.old_u[1:-1, 1:-1]).flat
        return np.sqrt(np.dot(v,v))


class LaplaceSolver:
    """A simple Laplacian solver that can use different schemes to
    solve the problem."""
    def __init__(self, grid, stepper='numpychecker'):
        self.grid = grid
        self.setTimeStepper(stepper)

    def slowTimeStep(self, dt=0.0):
        """Takes a time step using straight forward Python loops."""
        g = self.grid
        nx, ny = g.u.shape
        dx2, dy2 = g.dx**2, g.dy**2
        dnr_inv = 0.5/(dx2 + dy2)
        u = g.u

	err = 0.0;
        for offset in range(1,3):
          for i in range(1, ny-1):
              for j in range(1 + ( ( i + offset ) % 2), nx-1, 2):
                  tmp = u[j,i]
                  u[j,i] = ((u[j-1, i  ] + u[j+1, i  ])*dy2 +
                            (u[j  , i-1] + u[j  , i+1])*dx2)*dnr_inv
                  diff = u[j, i] - tmp
                  err += diff**2
                  
        return np.sqrt(err)
    
    def numpyCheckerLogicalTimeStep(self, dt=0.0):
        """Takes a time step using a NumPy expression."""
        g = self.grid
        dx2, dy2 = g.dx**2, g.dy**2
        dnr_inv = 0.5/(dx2 + dy2)
        u = g.u
        g.old_u = u.copy() # needed to compute the error.
        
        if self.count == 0:
            self.c = g.u[1:-1, 1:-1]
            self.n = g.u[:-2, 1:-1]
            self.s = g.u[2:, 1:-1]
            self.e = g.u[1:-1, :-2]
            self.w = g.u[1:-1, 2:]

            self.idxs = np.fromfunction(lambda x,y: ((x) + (y)) % 2, (self.c.shape))

        for rb in [0, 1]:
            self.c[self.idxs == rb] = ((self.n[self.idxs == rb] + self.s[self.idxs == rb])*dy2 +
                                       (self.e[self.idxs == rb] + self.w[self.idxs == rb])*dx2)*dnr_inv

        return g.computeError()

    def numpyTimeStep(self, dt=0.0):
        """Takes a time step using a NumPy expression."""
        g = self.grid
        dx2, dy2 = g.dx**2, g.dy**2
        dnr_inv = 0.5/(dx2 + dy2)
        u = g.u
        g.old_u = u.copy() # needed to compute the error.

        u[1:-1, 1:-1] = ((u[0:-2, 1:-1] + u[2:, 1:-1])*dy2 +
                         (u[1:-1,0:-2] + u[1:-1, 2:])*dx2)*dnr_inv


        return g.computeError()
        
    def numpyCheckerTimeStep(self, dt=0.0):
        """Takes a time step using a NumPy expression."""
        g = self.grid
        dx2, dy2 = g.dx**2, g.dy**2
        dnr_inv = 0.5/(dx2 + dy2)
        u = g.u
        g.old_u = u.copy() # needed to compute the error.

        
        # Do red calculation
	u[1:-1:2, 1:-1:2] = ((u[0:-2:2, 1:-1:2] + u[2::2, 1:-1:2])*dy2 +
                             (u[1:-1:2,0:-2:2] + u[1:-1:2, 2::2])*dx2)*dnr_inv
	u[2:-1:2, 2:-1:2] = ((u[1:-2:2, 2:-1:2] + u[3::2, 2:-1:2])*dy2 +
                             (u[2:-1:2,1:-2:2] + u[2:-1:2, 3::2])*dx2)*dnr_inv

        # Do black calculation
	u[1:-1:2, 2:-1:2] = ((u[0:-2:2, 2:-1:2] + u[2::2, 2:-1:2])*dy2 +
                             (u[1:-1:2,1:-2:2] + u[1:-1:2, 3::2])*dx2)*dnr_inv
	u[2:-1:2, 1:-1:2] = ((u[1:-2:2, 1:-1:2] + u[3::2, 1:-1:2])*dy2 +
                             (u[2:-1:2,0:-2:2] + u[2:-1:2, 2::2])*dx2)*dnr_inv
        
        return g.computeError()

    def openclCheckerTimeStep(self, dt=0.0):
        """
        Takes a time step using a PyOpenCL kernel based on inline C code from:
        http://www.scipy.org/PerformancePython
        The original method has been modified to use a red-black method that is
        more parallelizable.
        """
        nx, ny = self.grid.u.shape
        
        if self.count == 0:
            g = self.grid
            dx2, dy2 = g.dx**2, g.dy**2
            dnr_inv = 0.5/(dx2 + dy2)
            u = g.u
                        
            self.err = np.empty( (ny-2,), dtype=np.float32)
            self.ctx = cl.Context(dev_type=cl.device_type.GPU)

            self.queue = cl.CommandQueue(self.ctx)
      
            mf = cl.mem_flags
            self.u_buf = cl.Buffer(self.ctx, mf.READ_WRITE, u.nbytes)
            self.err_buf = cl.Buffer(self.ctx, mf.READ_WRITE, self.err.nbytes)
          
            cl.enqueue_write_buffer(self.queue, self.u_buf, u).wait()
          
            self.prg = cl.Program(self.ctx, """
            __kernel void lp2dstep( __global float *u, __global float *err, const uint stidx )
            {          
                float tmp, diff;
                int i = get_global_id(0) + 1;

                if ( stidx == 1 )
              	  err[i-1] = 0.0;
          
                for ( int j = 1 + ( ( i + stidx ) %% 2 ); j<( %(nx)d-1 ); j+=2 ) {
                    tmp = u[%(ny)d*j + i];
                    u[%(ny)d*j + i] = ((u[%(ny)d*(j-1) + i] + u[%(ny)d*(j+1) + i])*%(dy2)g +
                                       (u[%(ny)d*j + i-1] + u[%(ny)d*j + i + 1])*%(dx2)g)*%(dnr_inv)g;
                    diff = u[%(ny)d*j + i] - tmp;
                    err[i-1] += diff*diff;
                }
            }""" % { 'nx': nx,
                     'ny': ny,
                     'dx2': dx2,
                     'dy2': dy2,
                     'dnr_inv': dnr_inv } )
          
            try:
                self.prg.build()
            except:
                print "Error:"
                print self.prg.get_build_info(self.ctx.get_info(cl.context_info.DEVICES)[0], cl.program_build_info.LOG)

        # Enqueue red, black steps
        lp1evt = self.prg.lp2dstep(self.queue, ((ny-2),), None, self.u_buf, self.err_buf, np.uint32(1))
        cl.enqueue_wait_for_events(self.queue, [lp1evt])
        lp2evt = self.prg.lp2dstep(self.queue, ((ny-2),), None, self.u_buf, self.err_buf, np.uint32(2))
        cl.enqueue_wait_for_events(self.queue, [lp2evt])      

        # Get Updated Error Vector
        cl.enqueue_read_buffer(self.queue, self.err_buf, self.err).wait()
      
        return np.sqrt(np.sum(self.err))

    def openclCheckerFinish(self):
        cl.enqueue_read_buffer(self.queue, self.u_buf, self.grid.u).wait()

    def setTimeStepper(self, stepper='numpychecker'):
        """Sets the time step scheme to be used while solving given a
        string"""
        if stepper == 'slow':
            self.timeStep = self.slowTimeStep
            self.finish = None
        elif stepper == 'numpychecker':
            self.timeStep = self.numpyCheckerTimeStep
            self.finish = None
        elif stepper == 'numpylogicalchecker':
            self.timeStep = self.numpyCheckerLogicalTimeStep
            self.finish = None
        elif stepper == 'numpy':
            self.timeStep = self.numpyTimeStep
            self.finish = None
        elif stepper == 'openclchecker':
            self.timeStep = self.openclCheckerTimeStep
            self.finish = self.openclCheckerFinish
        else:
            self.timeStep = None

    def solve(self, n_iter=0, eps=1.0e-8):
        self.count = 0
        err = self.timeStep()
        self.count = 1

        while err > eps:
            if n_iter and self.count >= n_iter:
                if self.finish:
                    self.finish()
                return err
            err = self.timeStep()
            self.count += 1
        
        if self.finish:        
            self.finish()
        print "Converged!"
        return self.count
        

def BC(x, y):
    """Used to set the boundary condition for the grid of points.
    Change this as you feel fit."""    
    return (x**2 - y**2)


def test(nmin=5, nmax=30, dn=5, eps=1.0e-16, n_iter=0, stepper='numpychecker'):
    iters = []
    n_grd = numpy.arange(nmin, nmax, dn)
    times = []
    for i in n_grd:
        g = Grid(nx=i, ny=i)
        g.setBCFunc(BC)
        s = LaplaceSolver(g, stepper)
        t1 = time.clock()
        iters.append(s.solve(n_iter=n_iter, eps=eps))
        dt = time.clock() - t1
        times.append(dt)
        print "Solution for nx = ny = %d, took %f seconds"%(i, dt)
    return (n_grd**2, iters, times)


def time_test(nx=512, ny=512, eps=1.0e-16, n_iter=100, stepper='numpychecker'):
    g = Grid(nx, ny)
    g.setBCFunc(BC)
    s = LaplaceSolver(g, stepper)
    t = time.clock()
    s.solve(n_iter=n_iter, eps=eps)
    return time.clock() - t


def main(n=256, n_iter=100):
    print "Doing %d iterations on a %dx%d grid"%(n_iter, n, n)
    for i in [ 'numpychecker', 'openclchecker']:
        print i,
        sys.stdout.flush()
        t_elap = time_test(n, n, stepper=i, n_iter=n_iter)
        print ": %2.2f seconds" % (t_elap)

    print "slow (1 iteration)",
    sys.stdout.flush()
    s = time_test(n, n, stepper='slow', n_iter=1)
    print "took", s, "seconds"
    print "%d iterations should take about %f seconds"%(n_iter, s*n_iter)


if __name__ == "__main__":
    main()
