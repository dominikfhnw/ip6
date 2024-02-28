#!/usr/bin/env python3
import timey
import logging
import cv2 as cv

logging.basicConfig(
    #format='%(asctime)s: %(message)s',
    #format='%(asctime)s.%(msecs)03d %(levelname)s:%(module)s: %(message)s',
    format='%(asctime)s:%(module)s: %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
log = logging.info
dbg = logging.debug

def closed(str):
    return cv.getWindowProperty(str, cv.WND_PROP_VISIBLE) < 1

def res(x,y):
    log("change res")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, x)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, y)
    log("res changed")
log("start camera")
#list_ports()
#exit()
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
dbg("camera check")
if not cap.isOpened():
    logging.fatal("Cannot open camera")
    exit()

log("init time: "+timey.fromstr())
log("beginning main loop")

while True:
    t1 = timey.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        logging.fatal("Can't receive frame (stream end?). Exiting ...")
        break

    cv.imshow('frame', frame)

    match chr(cv.pollKey() & 0xFF):
        case 'q':
            break
        case '1':
            res(1280,720)
        case '2':
            res(640,480)
        case '3':
            res(10,10)

    if closed("frame"):
        break
    t2 = timey.time()
    #log(1/(t2 - t1 + 0.0001))

cv.destroyAllWindows()
log("total time: "+timey.fromstr())
