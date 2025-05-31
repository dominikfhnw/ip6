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
    global depth, ir
    ret = np.load("gesture/hand.npz")
    log(f"{ret=}")
    depth = ret["depth"]
    ir = ret["ir"]

    width = depth.shape[1]
    height = depth.shape[0]

    meta.set("width", width)
    meta.set("height",height)

    dbg("end init")
    return width, height

def get():
    if Color:
        color = np.zeros((height, width, 4), np.dtype('u1'))
    else:
        color = None
    return depth, ir, color