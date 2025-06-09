from time import perf_counter_ns
import atexit
import log

log, dbg, logger, isdbg = log.auto2(__name__)
# purpose of this module:
# - load it as first module, and note start time as soon as possible,
#   without linter complaining about code between imports
# - provide simple helper function to get time delta from start
# - provide time()

def time():
    return perf_counter_ns() / (10 ** 9)

t0 = time()

def delta(name, t2, end = None):
    if end is None:
        end = time()
    result = end - t2
    if isdbg:
       dbg(f"{name} time: {1000*result: .3f}ms")
    #log(f"{name} time: {1000*result: .3f}ms")
    return result

def fromstart():
    return time() - t0

def fromstr():
    return str(fromstart())

def _end():
    log("total time: "+fromstr())

atexit.register(_end)

class Timey(float):
    def __new__(cls, name):
        return super().__new__(cls, time())
    def __init__(self, name):
        self.name = name
    def delta(self, end = None):
        return delta(self.name, self, end)