import log
import cv2 as cv
import numpy as np
import glob
import os

log, dbg, _ = log.auto(__name__)
mtx = None

files = glob.glob("calibration/calib-*.npz")
files.sort(key=os.path.getmtime)
if len(files) > 0:
    calib = files[-1]
    log("calib file "+calib)
    #matrix = np.loadtxt(calib)
    #dbg(matrix)
    data = np.load(calib)
    mtx = data["mtx"]
    dist = data["dist"]
    shape = data["shape"]
    final_error = data["final_error"]
    log(dist)
    log(mtx)
    log("shape: "+str(shape)+", final_error: "+str(final_error))

else:
    log("no calibration file found")

def process(img):
    if mtx is None:
        return img
    return cv.undistort(img, mtx, dist, None, None)
