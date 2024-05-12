import log
import cv2 as cv
import numpy as np
from isave import isave
from math import floor, ceil
from imagefunctions import stabilize, avg, otsu
import segments
import meta

log, dbg, logger, isdbg = log.auto2(__name__)

LENGTH_REL = 3
HEIGHT = 300
AVG = 10
INTERPOLATION = cv.INTER_LINEAR


ROI_X = LENGTH_REL * HEIGHT
ROI_Y = HEIGHT
images = []
images_stab = []
last = None

def normalize(img):
    if meta.true("histNormalize"):
        return cv.equalizeHist(img)
    else:
        return img


def process(img, ids, corners):
    global images
    global images_stab
    global last

    debug = meta.get("extractDebug")

    if meta.get("key") == "r":
        log("RESET")
        images = []
        images_stab = []

    left, right = None, None
    for num, value in enumerate(ids):
        if value == 4:
            left = num
            dbg("found left:"+str(num))
        if value == 7:
            right = num
            dbg("found right:"+str(num))

    if left is None or right is None:
        return img

    dbg("found both")
    l = corners[left]
    r = corners[right]

    if isdbg: dbg("right: "+str(corners[right]))
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
    meta.set("res_point", (minx, maxy))
    roirect = img[miny:maxy, minx:maxx]
    if debug:
        cv.imshow("roirect", roirect)
    isave(roirect, "roi-rect")
    dbg(f"minmax: {minx},{miny} {maxx},{maxy}")

    a = (int(xa[0]), int(xa[1]))
    b = (int(xb[0]), int(xb[1]))
    c = (int(xc[0]), int(xc[1]))
    d = (int(xd[0]), int(xd[1]))

    pts1 = np.float32([xa,xb,xc,xd])
    # calculate offset relative to y-Axis, because we know the AruCo marker is y tall
    # and length in x direction changes with different displays
    offset = ROI_Y/10
    pts2 = np.float32([[-offset,0], [-offset,ROI_Y], [ROI_X+offset, 0], [ROI_X+offset, ROI_Y]])
    #pts2 = np.float32([[offset,0], [offset,ROI_Y], [ROI_X-offset, 0], [ROI_X-offset, ROI_Y]])

    M = cv.getPerspectiveTransform(pts1,pts2)
    # increasing dsize here will just copy more pixels from the original image
    dst = cv.warpPerspective(img,M,(ROI_X, ROI_Y),flags=INTERPOLATION)
    dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    #cv.imshow("GGGG",dst)
    #cv.waitKey(0)
    images.append(dst)

    if meta.true("stabilize"):
        dst3 = dst.copy()
        if len(images_stab) > 0:
            dst3 = stabilize(dst3, last)
        images_stab.append(dst3)
        cx2 = avg(images_stab, AVG)
        last = cx2
        if debug:
            ret, cx3 = otsu(cx2)
        cx2 = normalize(cx2)
        if debug:
            cv.imshow("composite stab", cx2)
            cv.imshow("composite stab otsu", cx3)

    c2 = avg(images, AVG)
    #log(c2)
    if debug:
        ret, c3 = otsu(c2)
    #log("OTSU thresh: "+str(ret))
    if meta.true("ocrComposite"):
        ocr = c2.copy()
    else:
        ocr = dst

    c2 = normalize(c2)
    if debug:
        cv.imshow("composite", c2)
        isave(c3,"composite-otsu")
        isave(c2, "composite")
        cv.imshow("composite otsu", c3)
    cnt = len(images)
    log("len "+str(cnt))

    isave(dst, "roi-gray")
    ret, dst2 = otsu(dst)
    dst = normalize(dst)

    if debug:
        cv.imshow("roi-thresh", dst2)
    isave(dst2, "roi-thresh")

    if debug:
        cv.imshow("ROI", dst)

    isave(img, "detect-raw")
    if meta.true("drawROI"):
        cv.line(img, a, b, (0,0,255), 5)
        cv.line(img, a, c, (0,0,255), 5)
        cv.line(img, b, d, (0,0,255), 5)
        cv.line(img, c, d, (0,0,255), 5)
        isave(img, "detect-marked")

    if meta.true("ocr"):
        segments.process(ocr, HEIGHT)
    return img