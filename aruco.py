import log
import cv2 as cv
import extract
import meta

log, dbg, logger = log.auto(__name__)

DICT = cv.aruco.DICT_4X4_50
PERIMETER = 0.01
# detect a bit smaller tags
REFINE = cv.aruco.CORNER_REFINE_CONTOUR
# NONE is a bit jumpy, APRILTAG is slow. CONTOUR only works well sometimes
# SUBPIX seems to be the best
# if cpu is an issue you can maybe choose between SUBPIX here or disable picture averaging in the
# extract module

arucoDict = cv.aruco.getPredefinedDictionary(DICT)
param = cv.aruco.DetectorParameters()
param.minMarkerPerimeterRate = PERIMETER
param.cornerRefinementMethod = REFINE
det = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=param)

def process(img):

    corners, ids, rejectedImgPoints = det.detectMarkers(img)
    if ids is not None:
        pass
        img = extract.process(img, ids, corners)
    if meta.true("drawAruco"):
        img = cv.aruco.drawDetectedMarkers(img, corners, ids)
    if meta.true("drawRejects"):
        img = cv.aruco.drawDetectedMarkers(img, rejectedImgPoints)

    return img
