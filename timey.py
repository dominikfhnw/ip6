from time import time
import atexit
import log

log, dbg, logger = log.auto(__name__)
# purpose of this module:
# - load it as first module, and note start time as soon as possible,
#   without linter complaining about code between imports
# - provide simple helper function to get time delta from start
# - provide time()

t0 = time()

def fromstart():
    return time() - t0

def fromstr():
    return str(fromstart())

def _end():
    log("total time: "+fromstr())

atexit.register(_end)