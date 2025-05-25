import sys
import socket
import cv2 as cv
from timestring import iso8601, timestring
import logging

# used like a singleton object

meta = dict(
    python = sys.version,
    opencv = cv.__version__,
    host = socket.gethostname(),
    platform = sys.platform,
    start = iso8601(),
    start_timestr = timestring(),
    frame = 0,
    ft = [0.033],
    drawAruco = False,
    drawRejects = False,
    drawROI = True,
    histNormalize = False,
    stabilize = False,          # failed stabilization attempt
    ocr = True,
    ocrComposite = True,        # frame averaging
    average = 10,
    thresh = False,
    logLevel = logging.INFO,
    invertDigits = True,
    extractDebug = True,
    height = 100,
    gui = True,
    hands = 1,
    gestures = True,
)

def get(name):
    return meta.get(name, None)

def getnum(name):
    return meta.get(name, 0)

def true(name):
    return bool(meta.get(name, False))

# assumes boolean True if no value was given
def setkey(name, val=True):
    global meta
    meta[name] = val

def setdict(dict):
    for key, val in dict.items():
        setkey(key, val)

# you can't set a key to None. use setkey or unset for that
def set(name, val=None):
    if type(name) is dict:
        if val is not None:
            raise Exception("dictionary with value")
        setdict(name)
    elif val is None:
        setkey(name, True)
    else:
        setkey(name, val)

def unset(name):
    setkey(name, None)

def toggle(name):
    global meta
    meta[name] = not meta[name]
    return meta[name]

def inc(name):
    global meta
    meta[name] = meta.get(name, 0) + 1

def append(name, val):
    global meta
    if not name in meta:
        meta[name] = []
    meta[name].append(val)