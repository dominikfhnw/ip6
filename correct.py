import log
import cv2 as cv
import numpy as np
import glob
import os

log, dbg, _ = log.auto(__name__)

files = glob.glob("calib-*.npz")
files.sort(key=os.path.getmtime)
calib = files[-1]
log("calib file "+calib)
#matrix = np.loadtxt(calib)
#dbg(matrix)
data = np.load(calib)
mtx = data["mtx"]
dist = data["dist"]

def process(img):
    return cv.undistort(img, mtx, dist, None, None)
