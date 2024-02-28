import log
import cv2 as cv

log, dbg, _ = log.auto(__name__)

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
arucoParams = cv.aruco.DetectorParameters()
#arucoParams.minMarkerPerimeterRate = 0.01
det = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=arucoParams)


def detect(img):
    #log("foo")
    corners, ids, rejectedImgPoints = det.detectMarkers(img)
    #print("CORNERS:",corners,", IDS:",ids)
    frame = cv.aruco.drawDetectedMarkers(img, corners, ids)
    frame = cv.aruco.drawDetectedMarkers(img, rejectedImgPoints)

    return img
