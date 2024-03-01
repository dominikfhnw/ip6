import log
import cv2 as cv
import extract

log, dbg, logger = log.auto(__name__)

DICT = cv.aruco.DICT_4X4_50

arucoDict = cv.aruco.getPredefinedDictionary(DICT)
p1 = cv.aruco.DetectorParameters()
p1.minMarkerPerimeterRate = 0.01
p2 = cv.aruco.DetectorParameters()
p2.minMarkerPerimeterRate = 0.01
p3 = cv.aruco.DetectorParameters()
p3.minMarkerPerimeterRate = 0.01
p4 = cv.aruco.DetectorParameters()
p4.minMarkerPerimeterRate = 0.01

p1.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
# REFINE_CONTOUR seems to "dance" around the least, especially at higher distances
# on second though... SUBPIX seems quite stable when near, and now also in lowres...
p2.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
# APRILTAG is quite slow
p3.cornerRefinementMethod = cv.aruco.CORNER_REFINE_APRILTAG
p4.cornerRefinementMethod = cv.aruco.CORNER_REFINE_NONE

#det1 = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=p1)
det2 = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=p1)
#det3 = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=p3)

def process(img, meta):

    corners, ids, rejectedImgPoints = det2.detectMarkers(img)
    if ids is not None:
        img = extract.process(img, ids, corners, meta)
    log("SHAPE IMG "+str(img.shape))
    if meta["drawAruco"]:
        img = cv.aruco.drawDetectedMarkers(img, corners, ids)
    if meta["drawRejects"]:
        img = cv.aruco.drawDetectedMarkers(img, rejectedImgPoints)

    return img
