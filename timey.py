from time import time

t0 = time()

def fromstart():
    return time() - t0

def fromstr():
    return str(fromstart())
