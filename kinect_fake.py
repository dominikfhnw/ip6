import meta
import numpy as np
import log

log, dbg, logger = log.auto(__name__)

Wide = meta.true("kinect_wide")
Color = meta.true("kinect_color")
Unbinned = meta.true("kinect_wide_unbinned")

def init():
    dbg("start init")

    global width, height
    width = 512
    height = 512

    meta.set("width", width)
    meta.set("height",height)

    dbg("end init")
    return width, height

def get():
    color = np.zeros((height, width, 4), np.dtype('u1'))
    depth = np.zeros((height, width), np.dtype('u2'))

    return depth, depth, color
