#!/usr/bin/env python3
import timey
import log
import aruco
import correct
import calib
import cv2 as cv
import pprint
from isave import isave
import segments
import meta

Camera = 0 # which camera to use
Mirror = False # flip image (for webcam)
Correct = True # apply camera calibration
Detect = True # detect and do stuff with aruco markers (the whole point here)
captureAPI = cv.CAP_ANY
if meta.get("platform") == "win32":
    # default API is really slow to start on windows
    # and does not support changing resolution on my machine
    captureAPI = cv.CAP_DSHOW

log, dbg, logger = log.auto(__name__)
# why oh why does the python logger not have a "notice" prio?
logger.warning("Seven segment display detector")

log("Python "+meta.get("python"))
log("OpenCV-Python "+meta.get("opencv"))
log("Host "+meta.get("host")+" on "+meta.get("platform"))
dbg("sysinfo done")

def closed(str):
    return cv.getWindowProperty(str, cv.WND_PROP_VISIBLE) < 1

def res(x,y):
    log("change res")
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, y)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, x)
    cap.set(cv.CAP_PROP_FPS, 10000)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    log(f"Changed to {width}x{height}, {fps}fps")

def showfps(img):
    framecount = 30
    ft = len(meta.get("ft"))
    if ft > framecount:
        ft = framecount
    fps = int(ft / sum(meta.get("ft")[-ft:]))
    cv.putText(img, f'{fps}fps',
               (10,30),
               fontFace=cv.FONT_HERSHEY_PLAIN,
               fontScale=1,
               color=(255, 255, 255)
               )

log("start camera")
cap = cv.VideoCapture(Camera, captureAPI)
dbg("camera check")
if not cap.isOpened():
    logger.fatal("Cannot open camera")
    exit()
dbg("camera opened")
fps = cap.get(cv.CAP_PROP_FPS)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
log(f"Camera {Camera}: {width}x{height}, {fps}fps")
log(f"Backend: {cap.getBackendName()}")
dbg("camera initialized")
meta.set(dict(
    backend = cap.getBackendName(),
    camera = Camera,
    width = int(width),
    height = int(height),
))

log("init time: "+timey.fromstr())
log("beginning main loop")
while True:
    t1 = timey.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        logger.fatal("Can't receive frame (stream end?). Exiting ...")
        break

    meta.unset("key")

    match chr(cv.pollKey() & 0xFF).lower():
        case 'q':
            break
        case '1':
            res(1280,720)
        case '2':
            res(640,480)
        case '3':
            res(10,10)
        case 'n':
            meta.toggle("histNormalize")
        case 'm':
            Mirror = not Mirror
        case 'c':
            Correct = not Correct
        case 'r':
            meta.set("key", "r")
        case 's':
            isave(frame, "frame")
            isave(out, "out")
        case ' ':
            out = calib.process(frame.copy())
            cv.imshow('out', out)
            cv.waitKey(500)

    if Mirror:
        copy = cv.flip(frame, 1)
    else:
        copy = frame.copy()
    showfps(copy)
    cv.imshow('frame',  copy)
    out = frame.copy()
    if Correct:
        out = correct.process(out)
    if Detect:
        out = aruco.process(out)
    cv.imshow('out', out)

    if closed("frame") or closed("out"):
        break

    meta.inc("frame")
    t2 = timey.time()
    meta.get("ft").append(t2 - t1 + 0.000001)
    #log(f'{1/(t2 - t1 + 0.0001):.0f}fps')

cv.destroyAllWindows()
log("total time: "+timey.fromstr())
