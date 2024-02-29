# code from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import numpy as np
import cv2 as cv
from timestring import timestring
import log
import atexit

log, dbg, _ = log.auto(__name__)

x = 9
y = 6

size = (x, y)

shape = None


def name():
    return "calibration/calib-" + timestring()


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x * y, 3), np.float32)
objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


def process(img):
    global objpoints
    global imgpoints
    global shape
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    shape = gray.shape[::-1]
    # cv.imshow('img',gray)
    # Find the chess board corners
    found, corners = cv.findChessboardCorners(gray, size, None, cv.CALIB_CB_FAST_CHECK)
    # If found, add object points, image points (after refining them)
    if found:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, size, corners2, found)
    return img


def finish():
    # log("done with pictures")
    num = len(objpoints)
    if num == 0:
        log("no calib data, returning")
        return
    log("Points: " + str(num))
    dbg(objpoints)
    dbg(imgpoints)
    file = name()
    np.savez_compressed(file + "-raw", imgpoints=imgpoints, objpoints=objpoints, shape=shape)
    log("saved raw datapoints")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, shape, None, None)
    log("got matrix")
    mean_error = 0
    for i in range(num):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    final_error = mean_error / len(objpoints)
    log("total error: {}".format(final_error))

    np.savez_compressed(file, mtx=mtx, dist=dist, final_error=final_error, shape=shape)
    log("finished writing calibration")


atexit.register(finish)
