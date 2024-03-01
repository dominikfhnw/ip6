import log
import cv2 as cv
import numpy as np
from isave import isave

log, dbg, logger = log.auto(__name__)

arucoDict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
param = cv.aruco.DetectorParameters()
#arucoParams.minMarkerPerimeterRate = 0.01
p1 = cv.aruco.DetectorParameters()
p2 = cv.aruco.DetectorParameters()
p3 = cv.aruco.DetectorParameters()
p1.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
# REFINE_CONTOUR seems to "dance" around the least, especially at higher distances
p2.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR
p3.cornerRefinementMethod = cv.aruco.CORNER_REFINE_NONE

det1 = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=p1)
det2 = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=p1)
det3 = cv.aruco.ArucoDetector(dictionary=arucoDict, detectorParams=p3)

images = []
composite = np.zeros(shape=(100,300), dtype=np.uint32)
composite2 = np.zeros(shape=(100,300), dtype=np.uint32)
composite3 = np.zeros(shape=(100,300), dtype=np.uint32)

last = None
#cv.imshow("composite", composite)

def phaseCorrelate(img1, img2):
    #log("SHAPE:"+str(img2.shape))
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret = cv.phaseCorrelate(np.float32(img1),np.float32(img2))
    log("correlation signal power: "+str(ret[1]))
    return ret[0]

def warpAffine(img, M, borderValue=(255,255,255)):
    return cv.warpAffine(img, M, (img.shape[1], img.shape[0]),borderValue=borderValue)

def translationMatrix(point):
    return np.array([
        [1, 0, point[0]],
        [0, 1, point[1]]
    ]).astype('float32')

def stabilize(img, reference):
    r = phaseCorrelate(img, reference)
    M = translationMatrix(r)
    return warpAffine(img, M)

# i1 = cv.imread("b1.png")
# i2 = cv.imread("b2.png")
# log("SHAPE "+str(i2.ndim))
# r = phaseCorrelate(i2,i1)
# log("SHAPE "+str(i2.ndim))
#
# M = translationMatrix(r)
# log("CORR "+str(r))
# M2 = np.array([
#     [1, 0, r[0]],
#     [0, 1, r[1]]
# ]).astype('float32')
# #i2w = warpAffine(i2, M)
# i2w = stabilize(i2, i1)
#
# cv.imshow("i1", i1)
# cv.imshow("i2", i2)
# cv.imshow("i2w", i2w)
# cv.imwrite("i2w.png", i2w)
# cv.waitKey(0)

def foo(img, ids, corners, name="ROI", meta=None):
    global images
    global composite
    global composite2
    global last


    left, right = None, None
    for num, value in enumerate(ids):
        if value == 4:
            left = num
            log(name+" found left:"+str(num))
        if value == 7:
            right = num
            log(name+" found right:"+str(num))

    if left is not None and right is not None:
        log(name+" found both")
        #log("left: "+str(left))
        #log("right: "+str(right))
        #log(name+" left: "+str(corners[left]))
        l = corners[left]
        r = corners[right]

        log(name+" right: "+str(corners[right]))
        xa = l[0][1]
        xb = l[0][2]
        xc = r[0][0]
        xd = r[0][3]

        a = (int(xa[0]), int(xa[1]))
        b = (int(xb[0]), int(xb[1]))
        c = (int(xc[0]), int(xc[1]))
        d = (int(xd[0]), int(xd[1]))

        pts1 = np.float32([xa,xb,xc,xd])
        pts2 = np.float32([[0,0], [0,100], [300,0], [300,100]])
        M = cv.getPerspectiveTransform(pts1,pts2)
        dst = cv.warpPerspective(img,M,(300,100))
        dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        images.append(dst)
        dst3 = dst.copy()
        if last is not None:
            tuple = phaseCorrelate(dst, last)
            log("PHASE: "+str(tuple))

            y = tuple[0]
            x = tuple[1]
            M = np.array([
                [1, 0, -y],
                [0, 1, -x]
            ]).astype('float32')
            dst3 = warpAffine(dst, M)

        #log("T1 "+str(dst.dtype))
        #log("T2 "+str(dst3.dtype))
        #og(composite.shape)
        #log(composite.dtype)
        cnt = len(images)
        log("len "+str(cnt))

        #composite = cv.add(composite, dst, composite, None, np.dtype(np.uint16).num)
        composite += dst
        composite2 += dst3
        #log(composite)
        c2 = composite.copy()
        c2 = (c2/cnt).astype('uint8')
        cv.imshow("composite", c2)

        cx2 = composite2.copy()
        cx2 = (cx2/cnt).astype('uint8')
        cv.imshow("composite3", cx2)


        ret, c3 = cv.threshold(c2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow("composite2", c3)

#c2 = cv.equalizeHist(c)
        cv.imshow("composite", c2)
        isave(dst, "roi-gray")
        #dst = cv.equalizeHist(dst)
        ret, dst2 = cv.threshold(dst,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow("roi-thresh", dst2)

        if dst3 is not None:
            cv.imshow("affine", dst3)
        cv.imshow(name, dst)

        isave(img, "detect-raw")
        cv.line(img, a, b, (0,0,255), 5)
        cv.line(img, a, c, (0,0,255), 5)
        cv.line(img, b, d, (0,0,255), 5)
        cv.line(img, c, d, (0,0,255), 5)

        isave(img, "detect-marked")
        isave(dst, "roi-thresh")
        last = dst
        #rect = [corners[left][2], corners[left][3], ]

def process(img, meta=None):
    if meta["key"] == "r":
        log("RESET")
        global images
        global composite
        global composite2
        images = []
        composite = np.zeros(shape=(100,300), dtype=np.uint32)
        composite2 = np.zeros(shape=(100,300), dtype=np.uint32)

#log("foo")
    #corners1, ids1, rejectedImgPoints1 = det1.detectMarkers(img)
    #if ids1 is not None:
    #    foo(img.copy(), ids1, corners1, "ROI1")

    corners2, ids2, rejectedImgPoints2 = det2.detectMarkers(img)
    if ids2 is not None:
        foo(img, ids2, corners2, "ROI2")

    #corners3, ids3, rejectedImgPoints3 = det3.detectMarkers(img)
    #if ids3 is not None:
    #    foo(img.copy(), ids3, corners3, "ROI3")

    #log(ids)
    #log(corners)
    #print("CORNERS:",corners,", IDS:",ids)
    frame = cv.aruco.drawDetectedMarkers(img, corners2, ids2)
    #frame = cv.aruco.drawDetectedMarkers(img, rejectedImgPoints)


    return img
