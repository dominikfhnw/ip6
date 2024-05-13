#!/usr/bin/env python3
import timey
import log
import aruco
import correct
import calib
import cv2 as cv
from isave import isave, ishow
import segments
import meta
import statistics
from timestring import timestring

SegDebug = True # debug segment part
SegDebug = False
Scale = 1 # global scale for small videos

Camera = 0 # which camera to use
AutoExposure = True # let camera do stuff VS settings for high fps
Mirror = False # flip image (for webcam)
Correct = False # apply camera calibration
Detect = True # detect and do stuff with aruco markers (the whole point here)
VideoWrite = False # write out video
#File = "video/vid20240302-155541-in.avi" # process this file if defined
#File = "rtsp://192.168.1.221:8080/h264.sdp" # rtsp streams also work

captureAPI = cv.CAP_ANY
if meta.get("platform") == "win32":
    # default API is really slow to start on windows
    # and does not support changing resolution on my machine
    captureAPI = cv.CAP_DSHOW
    #captureAPI = cv.CAP_MSMF

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
    cap.set(cv.CAP_PROP_FPS, 60)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    log(f"Changed to {width}x{height}, {fps}fps")

def showfps(img):
    framecount = 60
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

if SegDebug:
    segments.seg()

log("start camera")
if 'File' in vars() and bool(File):
    cap = cv.VideoCapture(File)
    VideoWrite=False
else:
    cap = cv.VideoCapture(Camera, captureAPI)
dbg("camera check")
if not cap.isOpened():
    logger.fatal("Cannot open camera")
    exit()
dbg("camera opened")
#fps = cap.get(cv.CAP_PROP_FPS)
#cap.set(cv.CAP_PROP_SHARPNESS,0) # disable any sharpening
#cap.set(cv.CAP_PROP_SETTINGS, 1)
#if AutoExposure:
#    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    #cap.set(cv.CAP_PROP_BRIGHTNESS, 0)
    #cap.set(cv.CAP_PROP_CONTRAST, 0)
#else:
cap.set(cv.CAP_PROP_EXPOSURE, -8)
    #cap.set(cv.CAP_PROP_BRIGHTNESS, 100) # artificially boost brightness
    #cap.set(cv.CAP_PROP_CONTRAST,8)
    #cap.set(cv.CAP_PROP_FOCUS, 180)
#log("CAM SETTING "+str(cap.set(cv.CAP_PROP_CONTRAST,0)))
#PROP = cv.CAP_PROP_CONTRAST # only sw

#PROP = cv.CAP_PROP_AUTO_EXPOSURE # seems to get autodisabled as soon as we set _EXPOSURE
PROP = cv.CAP_PROP_EXPOSURE # range: -8 to -4 (for acceptable framerates
#PROP = cv.CAP_PROP_BRIGHTNESS # only sw

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
log(f"Camera {Camera}: {width}x{height}")
log(f"Backend: {cap.getBackendName()}")
dbg("camera initialized")
meta.set(dict(
    camerabackend = cap.getBackendName(),
    camera = Camera,
    camerawidth = int(width),
    cameraheight = int(height),
))

#res(352,288)
if VideoWrite:
    # TODO: resolution
    vout1 = cv.VideoWriter("video/vid"+timestring()+"-in.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30.0, (width,height))
    vout2 = cv.VideoWriter("video/vid"+timestring()+"-out.avi", cv.VideoWriter_fourcc('M','J','P','G'), 30.0, (width,height))

Exposure = -8
log("init time: "+timey.fromstr())
log("beginning main loop")
gui = meta.get("gui")
while True:
    t1 = timey.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        logger.fatal("Can't receive frame (stream end?). Exiting ...")
        break
    if VideoWrite:
        vout1.write(frame)
    #log("FOCUS "+str(cap.get(cv.CAP_PROP_FOCUS)))
    meta.unset("key")
    meta.unset("save")
    meta.unset("result")

    match chr(cv.pollKey() & 0xFF).lower():
        case 'q':
            break
        case '1':
            res(1280,720)
            Scale = 1
        case '2':
            res(640,480)
            Scale = 1
        case '3':
            res(10,10)
            Scale = 2
        case 'n':
            meta.toggle("histNormalize")
        case 'm':
            Mirror = not Mirror
        case 'c':
            Correct = not Correct
        case 'i':
            cv.waitKey(0)
        case 'p':
            cap.set(cv.CAP_PROP_SETTINGS, 1)
        case 'a':
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
        case ',':
            Exposure -= 1
            cap.set(PROP, Exposure)
            log("Exposure: "+str(Exposure))
        case '.':
            Exposure += 1
            cap.set(PROP, Exposure)
            log("Exposure: "+str(Exposure))
        case 's':
            meta.toggle("stabilize")
        case 'x':
            meta.toggle("ocrComposite")
        case 't':
            meta.toggle("thresh")
        case 'r':
            meta.set("key", "r")
        case 'w':
            meta.set("save")
            log("save requested")
            isave(frame, "frame",True)
            isave(out, "out", True)
        case ' ':
            out = calib.process(frame.copy())
            ishow('out', out)
            cv.waitKey(500)

    copy = frame.copy()
    if gui and Scale != 1:
        copy = cv.resize(copy, None, None, Scale, Scale, cv.INTER_NEAREST)
    out = copy.copy()
    if gui and Mirror:
        copy = cv.flip(copy, 1)
    showfps(copy)
    ishow('frame',  copy)
    #out = frame - np.min(frame)
    #out = (out * (256/np.max(out))).astype("uint8")
    if Correct:
        out = correct.process(out)
    if Detect:
        out = aruco.process(out)

    result = meta.get("result")
    if result is not None:
        color=(0,255,0)
        if "REJ" in result:
            log("REJ found")
            color=(0,0,255)
            result=result.replace("REJ", "")
        log(f"res {result=}")
        point = meta.get("res_point")
        p2 = (point[0]+20, point[1]+40)
        # TODO: separate function
        cv.putText(out, result,
           p2,
           fontFace=cv.FONT_HERSHEY_DUPLEX,
           fontScale=1,
           color=color
        )
    showfps(out)
    if VideoWrite:
            vout2.write(out)
    ishow('out', out)

    if gui and (closed("frame") or closed("out")):
        break

    meta.inc("frame")
    t2 = timey.time()
    meta.get("ft").append(t2 - t1 + 0.000001)
    #log(f'{1/(t2 - t1 + 0.0001):.0f}fps')

cv.destroyAllWindows()
