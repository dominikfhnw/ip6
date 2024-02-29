from time import time
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
