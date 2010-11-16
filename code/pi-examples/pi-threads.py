from threading import Thread, Lock
import random

lock = Lock() # lock for making operations atomic

def calcInside(nsamples,rank):
    global inside # we need something everyone can share
    random.seed(rank)
    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if (x*x)+(y*y)<1:
            lock.acquire() # GIL doesn't always save you
            inside += 1
            lock.release()

if __name__ == '__main__':
    nt=4 # thread count
    inside = 0 # you need to initialize this
    samples=int(12e6/nt)
    threads=[Thread(target=calcInside, args=(samples,i)) for i in range(nt)]
    
    for t in threads: t.start()
    for t in threads: t.join()

    print (4.0*inside)/(1.0*samples*nt)
