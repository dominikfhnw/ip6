import sys
import socket
import cv2 as cv
from timestring import iso8601
import logging

# used like a singleton object

meta = dict(
    python = sys.version,
    opencv = cv.__version__,
    host = socket.gethostname(),
    platform = sys.platform,
    start = iso8601(),
    frame = 0,
    ft = [0.033],
    drawAruco = True,
    drawRejects = True,
    drawROI = True,
    histNormalize = False,
    stabilize = False,
    ocr = True,
    ocrComposite = True,
    thresh = False,
    logLevel = logging.DEBUG,
    invertDigits = True,
    extractDebug = False,
)

def get(name):
    return meta[name]

def true(name):
    return bool(meta[name])

# assumes boolean True if no value was given
def setkey(name, val=True):
    global meta
    meta[name] = val

def setdict(dict):
    global meta
    for key, val in dict.items():
        setkey(key, val)

# you can't set a key to None. use setkey or unset for that
def set(name, val=None):
    if type(name) is dict:
        if val is not None:
            raise Exception("dictionary with value")
        setdict(name)
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
    meta[name] += 1
