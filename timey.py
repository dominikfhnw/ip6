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
    if not isdbg:
        return
    if end is None:
        end = time()
    dbg(f"{name} time: {1000*(end - t2): .3f}ms")

def fromstart():
    return time() - t0

def fromstr():
    return str(fromstart())

def _end():
    log("total time: "+fromstr())

atexit.register(_end)