import log
import cv2 as cv
import numpy as np
from isave import isave
from math import floor, ceil

log, dbg, logger = log.auto(__name__)

ROI_X = 300
ROI_Y = 100
AVG = 10
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

images = []
images_stab = []
composite = np.zeros(shape=(ROI_Y, ROI_X), dtype=np.uint32)
composite2 = np.zeros(shape=(ROI_Y, ROI_X), dtype=np.uint32)
composite3 = np.zeros(shape=(ROI_Y, ROI_X), dtype=np.uint32)

last = None
#cv.imshow("composite", composite)

def avg(image, count):
    composite = np.zeros(shape=(ROI_Y,ROI_X), dtype=np.uint32)
    last = image[-count:]
    for i in last:
        composite += i
    return (composite/count).astype('uint8')


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

def otsu(img):
    return cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

def foo(img, ids, corners, name="ROI", meta=None):
    global images
    global images_stab
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

        minx, miny, maxx, maxy = float('inf'), float('inf'), 0, 0
        for t in xa, xb, xc, xd:
            if t[0] < minx:
                minx = floor(t[0])
            if t[1] < miny:
                miny = floor(t[1])
            if t[0] > maxx:
                maxx = ceil(t[0])
            if t[1] > maxy:
                maxy = ceil(t[1])
        roirect = img[miny:maxy, minx:maxx]
        cv.imshow("roirect", roirect)
        isave(roirect, "roi-rect")
        log(f"minmax: {minx},{miny} {maxx},{maxy}")

        a = (int(xa[0]), int(xa[1]))
        b = (int(xb[0]), int(xb[1]))
        c = (int(xc[0]), int(xc[1]))
        d = (int(xd[0]), int(xd[1]))

        pts1 = np.float32([xa,xb,xc,xd])
        offset = 10 # TODO: yanky
        pts2 = np.float32([[-offset,0], [-offset,ROI_Y], [ROI_X+offset, 0], [ROI_X+offset, ROI_Y]])
        M = cv.getPerspectiveTransform(pts1,pts2)
        border=0
        dst = cv.warpPerspective(img,M,(ROI_X, ROI_Y),flags=cv.INTER_LINEAR)
        dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
        images.append(dst)
        dst3 = dst.copy()
        preavg = avg(images_stab, AVG)

        #if last is not None:
        if len(images_stab) > 0:
            #dst3 = stabilize(dst3, composite)
            dst3 = stabilize(dst3, preavg)
        images_stab.append(dst3)


        #log("T1 "+str(dst.dtype))
        #log("T2 "+str(dst3.dtype))
        #og(composite.shape)
        #log(composite.dtype)
        cnt = len(images)
        log("len "+str(cnt))

        #composite = cv.add(composite, dst, composite, None, np.dtype(np.uint16).num)
        #composite += dst
        #composite2 += dst3
        #log(composite)
        #c2 = composite.copy()
        #c2 = (c2/cnt).astype('uint8')
        c2 = avg(images, AVG)
        cv.imshow("composite", c2)
        ret, c3 = otsu(c2)
        isave(c3,"composite-otsu")
        isave(c2, "composite")
        cv.imshow("composite otsu", c3)

        #cx2 = composite2.copy()
        #cx2 = (cx2/cnt).astype('uint8')
        cx2 = avg(images_stab, AVG)
        cv.imshow("composite stab", cx2)
        ret, cx3 = otsu(cx2)
        cv.imshow("composite stab otsu", cx3)


#c2 = cv.equalizeHist(c)
        #cv.imshow("composite", c2)
        isave(dst, "roi-gray")
        #dst = cv.equalizeHist(dst)
        ret, dst2 = otsu(dst)
        cv.imshow("roi-thresh", dst2)

        #if dst3 is not None:
            #cv.imshow("affine", dst3)
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
        global images_stab
        global composite
        global composite2
        images = []
        images_stab = []
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
    #frame = cv.aruco.drawDetectedMarkers(img, corners2, ids2)
    #frame = cv.aruco.drawDetectedMarkers(img, rejectedImgPoints)


    return img
