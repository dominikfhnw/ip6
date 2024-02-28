import cv2 as cv
from log import logger
logger.info("blub")
logger = logger.getLogger(__name__)
log = logger.info
dbg = logger.debug
log("initialized detector module")

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
arucoParams = cv.aruco.DetectorParameters()
arucoParams.minMarkerPerimeterRate = 0.01
det = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=arucoParams)
log("finished init")
def detect(img):
    #log("foo")
    corners, ids, rejectedImgPoints = det.detectMarkers(img)
    #print("CORNERS:",corners,", IDS:",ids)
    frame = cv.aruco.drawDetectedMarkers(img, corners, ids)
    #frame = cv.aruco.drawDetectedMarkers(frame, rejectedImgPoints)

    return img