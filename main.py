#!/usr/bin/env python3
import timey
import log
import aruco
import correct
import calib
import cv2 as cv
import numpy as np
from isave import isave, ishow
import segments
import meta
import statistics
import gestures
from timestring import timestring
import kinect

SegDebug = False # debug segment part
Scale = 1 # global scale for small videos
Scale2 = 1 # scale for final output

Camera = -1 # which camera to use
AutoExposure = True # let camera do stuff VS settings for high fps
Mirror = True # flip image (for webcam)
Correct = False # apply camera calibration
Aruco = False # detect and do stuff with aruco markers
VideoWrite = False # write out video
Fast = False # show images without processing

meta.set("ocrComposite", False)
#File = "benchmark/classroom.avi"
#File = "benchmark/highmotion.avi"
#File = "benchmark/smallanddark.avi"
#File = "benchmark/noglare.avi"
#File = "benchmark/noglare_exposuremotion.avi"

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

def getres():
    global width, height, fps
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv.CAP_PROP_FPS))
    meta.set("width", width)
    meta.set("height",height)
    meta.set("fps", fps)


def res(x,y, fps=30):
    log("change res")
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, y)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, x)
    cap.set(cv.CAP_PROP_FPS, fps)
    getres()
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

if Camera < 0:
    width, height = kinect.init()
else:
    log("start camera")
    start = meta.get("start")
    avgoption = meta.get("ocrComposite")
    source = f"Live {start }"
    meta.set("filename","LIVE")
    if 'File' in vars() and bool(File):
        cap = cv.VideoCapture(File)
        VideoWrite=False
        meta.set("filename", File)
        source = "File: "+File
    else:
        cap = cv.VideoCapture(Camera, captureAPI)
    dbg("camera check")
    if not cap.isOpened():
        logger.fatal("Cannot open camera")
        exit()
    dbg("camera opened")

    getres()
    PROP = cv.CAP_PROP_EXPOSURE # range: -8 to -4 (for acceptable framerates
    #PROP = cv.CAP_PROP_BRIGHTNESS # only sw
    #res(640,480)

    #res(10,10)
    log(f"Camera {Camera}: {width}x{height}, {fps}fps")
    log(f"Backend: {cap.getBackendName()}")
    dbg("camera initialized")

    meta.set("source", f"{source} AVG:{avgoption} {Camera}: {width}x{height}")
    meta.set(dict(
        camerabackend = cap.getBackendName(),
        camera = Camera,
        camerawidth = int(width),
        cameraheight = int(height),
    ))

#res(352,288)
if VideoWrite:
    # TODO: resolution
    vout1 = cv.VideoWriter("video-gest/vid"+timestring()+"-in.avi", cv.VideoWriter_fourcc('M','J','P','G'), 12, (height,height))
    vout2 = cv.VideoWriter("video-gest/vid"+timestring()+"-out.avi", cv.VideoWriter_fourcc('M','J','P','G'), 12, (height,height))

Exposure = -8
log("init time: "+timey.fromstr())
log("beginning main loop")
gui = meta.get("gui")

if Fast:
    #res(10,10,120)
    #res(4096,3072,15)
    #res(2048,1536,30)

    while True:
        t1 = timey.time()
        #ret, frame = cap.read()

        depth, ir = kinect.get()
        frame = cv.normalize(ir, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        match chr(cv.pollKey() & 0xFF).lower():
            case 'q':
                break
        showfps(frame)
        cv.imshow('out',frame)
        t2 = timey.time()
        meta.get("ft").append(t2 - t1 + 0.000001)
    exit(0)

while True:
    dbg("mainloop")
    t1 = timey.time()
    if Camera > 0:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            logger.fatal("Can't receive frame (stream end?). Exiting ...")
            break
    else:
        frame = kinect.process()
    # square image for hand detection
    offset = int((width-height)/2)
    frame = np.array(frame[0:height, offset:height+offset])
    if VideoWrite:
        vout1.write(frame)
    #log("FOCUS "+str(cap.get(cv.CAP_PROP_FOCUS)))
    meta.unset("key")
    meta.unset("save")
    #meta.unset("result")

    match chr(cv.pollKey() & 0xFF).lower():
        case 'q':
            break
        case '1':
            res(2000,1500)
            Scale = 1
            Scale2 = 1
        case '2':
            res(640,480)
            Scale = 1
            Scale2 = 1
        case '3':
            res(10,10)
            Scale = 2
            Scale2 = 1
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
            log("skeleton toggled")
            meta.toggle("skeleton")
        case 'e':
            log("ET toggled")
            meta.toggle("et")
        case 'l':
            log("lightsaber toggled")
            meta.toggle("lightsaber")


        case 'x':
            meta.toggle("ocrComposite")
        case 't':
            meta.toggle("thresh")
        case 'r':
            meta.set("key", "r")
        case 'w':
            meta.set("save")
            log("save requested")
            #isave(frame, "frame",True)
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
    #ishow('frame',  copy)
    #out = frame - np.min(frame)
    #out = (out * (256/np.max(out))).astype("uint8")
    if Correct:
        out = correct.process(out)
    if Aruco:
        out = aruco.process(out)
    if meta.true("mediapipe"):
        out = gestures.process(frame, t1)
    result = meta.get("result")
    if result is not None:
        color=(0,255,0)
        if "REJ" in result:
            color=(0,0,255)
            result=result.replace("REJ", "")
        point = meta.get("res_point")
        p2 = (point[0]+0, point[1]+0)
        # TODO: separate function
        cv.putText(out, result,
           point,
           fontFace=cv.FONT_HERSHEY_DUPLEX,
           fontScale=0.5,
           color=color
        )
    if gui and Scale2 != 1:
        out = cv.resize(out, None, None, Scale2, Scale2, cv.INTER_CUBIC)
    showfps(out)
    if VideoWrite:
            vout2.write(out)
    ishow('out', out)

    if gui and (closed("out")):
        dbg("window closed, breaking")
        break

    meta.inc("frame")
    t2 = timey.time()
    meta.get("ft").append(t2 - t1 + 0.000001)
    #log(f'{1/(t2 - t1 + 0.0001):.0f}fps')

cv.destroyAllWindows()
