import numpy as np
import cv2 as cv
from isave import ishow
import log
import meta
if meta.true("kinect_enable"):
    import kinect_raw
else:
    import kinect_fake as kinect_raw

log, dbg, logger = log.auto(__name__)


def init():
    return kinect_raw.init()


def get():
    return kinect_raw.get()


def process():
    depth_raw, ir, rgba = kinect_raw.get()
    if depth_raw is None:
        return np.zeros((meta.num("height"), meta.num("width"), 3), np.dtype('u1'))

    # Select distance range
    lo = np.array([meta.num("kinect_lo")])
    hi = np.array([meta.num("kinect_hi")])
    mask = cv.inRange(depth_raw, lo, hi)

    dr2 = cv.normalize(depth_raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
    dr2 = cv.applyColorMap(dr2, cv.COLORMAP_JET)
    dr2[mask == 0] = (127, 127, 127)
    ishow("dr2", dr2)

    # log(f"IR MAX: {minLoc=}={minVal} {maxLoc=}={maxVal}")
    if ir is not None:
        # fix the ultra bright "too much reflection" pixels
        ir[ir == 65535] = (ir.max() + 1)
        ir2 = ir.copy()

        ir = cv.normalize(ir, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        ireq = ir.copy()
        ir = cv.cvtColor(ir, cv.COLOR_GRAY2BGR)

        ir2 = cv.normalize(ir2, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
        ircolor = cv.cvtColor(ir2, cv.COLOR_GRAY2BGR)
        frame = ircolor.copy()

        minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(depth_raw, mask)
        #log(f"XXXXXXX {ircolor.shape=} {ircolor.dtype=} {depth_raw.shape=} {depth_raw.dtype=}")
        ircolor[depth_raw == maxVal]=(0,0,255)
        ircolor[depth_raw == minVal]=(255,0,0)

        cv.circle(ircolor, center=minLoc, radius=3, color=(255, 0, 0), thickness=-1)
        cv.circle(ircolor, center=maxLoc, radius=3, color=(0, 0, 255), thickness=-1)
        # cv.line(anno, pt1=index, pt2=base, thickness=2, color=(0, 165, 255))
        cv.putText(ircolor, f"{minVal / 10: .1f}cm",
                   (minLoc[0] + 10, minLoc[1]), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 255), 1, cv.LINE_AA)
        cv.putText(ircolor, f"{maxVal / 10: .1f}cm",
                   (maxLoc[0] + 10, maxLoc[1]), cv.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 255), 1, cv.LINE_AA)

    else:
        log("IR failed")
        exit(12)

    if rgba is not None:
        rgb = cv.cvtColor(rgba, cv.COLOR_BGRA2BGR)
        rgb2 = rgb.copy()
        rgb2[mask == 0] = (0, 0, 0)
        ishow("rgb2", rgb2)
        ishow("color", rgb)

        if meta.true("kinect_composite"):
            rgb3 = rgb.copy()
            black = np.array([0,0,0])
            mask2 = cv.inRange(rgb3, black, black)
            locs = np.where(mask2 != 0)
            ishow("mask2",mask2)
            img1 = cv.cvtColor(cv.equalizeHist(ireq), cv.COLOR_GRAY2BGR)
            ishow("img1",img1)
            rgb3[locs[0], locs[1]] = img1[locs[0], locs[1]]
            #rgb.Mat().copyTo(composite, mask2)
            ishow("composite", rgb3)

    if meta.true("kinect_depth"):
        depth = depth_raw.copy()
        #depth[depth>4000] = 0
        max=3000    # max distance in mm
        depth = cv.convertScaleAbs(depth, None, 255/max)
        #depth = cv.normalize(depth_raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        depth = cv.applyColorMap(depth, cv.COLORMAP_JET)
        #log(f"{depth_raw.min()=} {depth_raw.max()=}")
        ishow("depth", depth)
    ishow("ir", ir)
    #if meta.true("kinect_irnorm"):
    #ishow("ir2", ir2)
    ishow("ircolor", ircolor)
    # ishow("depth",depth)

    return frame
