#!/usr/bin/env python3
import timey
from log import logger
print(timey.fromstart())
import cv2 as cv
print(timey.fromstart())
import detect
import os
import socket
import sys

log = logger.info
dbg = logger.debug

Camera = 0 # which camera to use
Mirror = False # flip image (for webcam)
if os.name == "nt":
    # default API is really slow to start on windows
    # and does not support changing resolution
    captureAPI = cv.CAP_DSHOW
else:
    captureAPI = cv.CAP_ANY



log("Seven segment display detector")
#print("Python "+platform.python_version())
log("Python "+sys.version)
log("OpenCV "+cv.__version__)
#print("Platform "+platform.platform())
log("Host "+socket.gethostname()+" on "+os.name)
# platform is too slow
#print("Host "+platform.node()+" on "+platform.system())
dbg("sysinfo done")

def closed(str):
    return cv.getWindowProperty(str, cv.WND_PROP_VISIBLE) < 1

def res(x,y):
    log("change res")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, x)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, y)
    log("res changed")

def list_ports():
    """
    Test the ports and returns a tuple with the available ports and the ones that are working.
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []
    while len(non_working_ports) < 6: # if there are more than 5 nonworking ports stop the testing.
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports,non_working_ports


log("start camera")
#list_ports()
#exit()
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

    out = detect.detect(frame.copy())

    if Mirror:
        frame = cv.flip(frame, 1)

    cv.imshow('frame', frame)
    #cv.imshow('out', out)


    match chr(cv.pollKey() & 0xFF):
        case 'q':
            break
        case '1':
            res(1280,720)
        case '2':
            res(640,480)
        case '3':
            res(10,10)

    if closed("frame"):# or closed("out"):
        break
    t2 = timey.time()
    #log(1/(t2 - t1 + 0.0001))

cv.destroyAllWindows()
log("total time:"+timey.fromstr())
