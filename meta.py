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
    logLevel = logging.INFO,
    gui = True,
# seven-segment OCR options
    drawAruco = False,
    drawRejects = False,
    drawROI = True,
    histNormalize = False,
    stabilize = False,          # failed stabilization attempt
    ocr = True,
    ocrComposite = True,        # frame averaging
    average = 10,
    thresh = False,
    invertDigits = True,
    extractDebug = True,
    height = 100,
# Gesture recognition options
    mediapipe = True,       # global flag to load any mediapipe code
    mediapipe_gpu = False,  # GPU acceleration
    hands = 1,
    gestures = True,
    async_gestures = False,
    skeleton = True,
    et = False,
    lightsaber = False,
    hand_detection = 0.5,       # finding palm of hand
    hand_presence = 0.5,        # score to retrigger palm finding
    kinect_enable = True,       # enable Kinect
    kinect_color = False,       # also get color picture from kinect
    kinect_wide = False,        # wide FOV
    kinect_lo = 1,              # Select pixels that are between lo and hi
    kinect_hi = 600,            #   millimeters from the Kinect sensor
    kinect_step = 100,
    kinect_composite = False,   # DEBUG: composite color/IR image
    kinect_depth = False,       # DEBUG: show depth overview
    kinect_passive = False,     # DEBUG: passive IR only
)

def get(name):
    return meta.get(name, None)

def num(name) -> int:
    return int(meta.get(name, 0))

def true(name) -> bool:
    return bool(meta.get(name, False))

# assumes boolean True if no value was given
def setkey(name, val):
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

def add(name, val):
    meta[name] = meta[name] + val
    return meta[name]

def sub(name, val):
    meta[name] = meta[name] - val
    return meta[name]

def inc(name):
    global meta
    meta[name] = meta.get(name, 0) + 1

def append(name, val):
    global meta
    if not name in meta:
        meta[name] = []
    meta[name].append(val)