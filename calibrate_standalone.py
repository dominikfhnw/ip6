# code from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
import numpy as np
import cv2 as cv
import glob
from datetime import datetime
import log

log, dbg, _ = log.auto(__name__)

log("calibration start")
x = 8
y = 5

size = (x,y)

def name():
    return "calib-"+datetime.now().strftime("%Y%m%d-%H%M%S")

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x*y,3), np.float32)
objp[:,:2] = np.mgrid[0:x,0:y].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('out/*.jpg')
log(images)
log("start")
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('img',gray)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, size, None)
    print(str(ret)+" "+fname)
    #print(corners)
    #cv.waitKey(500)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1)

cv.destroyAllWindows()
cv.waitKey(1)
log("done with pictures")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
log("got matrix")
img = cv.imread(images[1])
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
log("optimal matrix")
#cv.destroyAllWindows()
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite(name()+".jpg", dst)
log("applied  matrix to first picture")
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
log( "total error: {}".format(mean_error/len(objpoints)) )

np.savez_compressed(name(), mtx=mtx, dist=dist, mtx2=newcameramtx)
#np.savetxt(name()+".txt", mtx=mtx, dist=dist)