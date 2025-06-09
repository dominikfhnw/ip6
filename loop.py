import log
import timey
import cv2 as cv
import meta
import kinect_raw
import kinect
import gestures
from time import sleep
from timey import Timey

log, dbg, logger = log.auto(__name__)

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


def gui(out):
    t1 = Timey("gui")
    out = cv.resize(out, None, None, 3, 3, cv.INTER_NEAREST)
    showfps(out)
    cv.imshow('out fast', out)
    match chr(cv.pollKey() & 0xFF).lower():
        case 'q':
            return False
    t1.delta()
    return True

def run():
    while True:
        t1 = Timey("loop")

        depth = kinect_raw.getdepth()
        if depth is None:
            log("no depth, restarting loop")
            t1.delta()
            continue

        frame = kinect.proc_fast(depth)

        out, result = gestures.proc_fast(frame, t1)
        #out = frame

        if not gui(out):
            break

        meta.append("ft", t1.delta())
    log("end fastloop")