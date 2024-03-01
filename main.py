#!/usr/bin/env python3
import timey
import log
import aruco
import correct
import calib
import cv2 as cv
import socket
import sys
import pprint
from timestring import timestring, iso8601
from isave import isave

Camera = 0 # which camera to use
Mirror = False # flip image (for webcam)
Correct = True # apply camera calibration
captureAPI = cv.CAP_ANY
if sys.platform == "win32":
    # default API is really slow to start on windows
    # and does not support changing resolution on my machine
    captureAPI = cv.CAP_DSHOW

log, dbg, logger = log.auto(__name__)
# why oh why does the python logger not have a "notice" prio?
logger.warning("Seven segment display detector")

log("Python "+sys.version)
log("OpenCV-Python "+cv.__version__)
#print("Platform "+platform.platform())
log("Host "+socket.gethostname()+" on "+sys.platform)
dbg("sysinfo done")

def closed(str):
    return cv.getWindowProperty(str, cv.WND_PROP_VISIBLE) < 1

def res(x,y):
    log("change res")
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, y)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, x)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    log(f"Changed to {width}x{height}, {fps}fps")

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
meta = dict(
    python = sys.version,
    opencv = cv.__version__,
    host = socket.gethostname(),
    platform = sys.platform,
    start = iso8601(),
    backend = cap.getBackendName(),
    camera = Camera,
    width = int(width),
    height = int(height),
    frame = 0,
    drawAruco = True,
    drawRejects = False,
    drawROI = True,
)
pp = pprint.PrettyPrinter(sort_dicts=False).pformat
#pp(meta)
log(pp(meta))

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

    meta["key"] = None

    match chr(cv.pollKey() & 0xFF):
        case 'q':
            break
        case '1':
            res(1280,720)
        case '2':
            res(640,480)
        case '3':
            res(10,10)
        case 'm':
            Mirror = not Mirror
        case 'c':
            Correct = not Correct
        case 'r':
            meta["key"] = "r"
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
        copy = frame

    cv.imshow('frame',  copy)
    out = frame.copy()
    if Correct:
        out = correct.process(out)
    out = aruco.process(out, meta)
    cv.imshow('out', out)

    if closed("frame") or closed("out"):
        break
    t2 = timey.time()
    #log(f'{1/(t2 - t1 + 0.0001):.0f}fps')
    meta["frame"] += 1

cv.destroyAllWindows()
log("total time: "+timey.fromstr())
