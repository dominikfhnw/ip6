import numpy as np
import cv2 as cv
import kinect_raw
from isave import ishow
import log
import meta

log, dbg, logger = log.auto(__name__)


def init():
    return kinect_raw.init()


def get():
    return kinect_raw.get()


def process():
    depth_raw, ir, color = kinect_raw.get()
    if depth_raw is None:
        return np.zeros((meta.num("height"), meta.num("width"), 3), np.dtype('u1'))

    # Select distance range
    lo = np.array([1])
    hi = np.array([500])
    mask = cv.inRange(depth_raw, lo, hi)

    dr2 = cv.normalize(depth_raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
    dr2 = cv.applyColorMap(dr2, cv.COLORMAP_JET)
    dr2[mask == 0] = (127, 127, 127)
    ishow("dr2", dr2)

    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(depth_raw, mask)
    # log(f"IR MAX: {minLoc=}={minVal} {maxLoc=}={maxVal}")
    if ir is not None:
        ir2 = ir.copy()

        # fix the ultra bright "too much reflection" pixels
        ir[ir == 65535] = (ir.max() + 1)

        ir = cv.normalize(ir, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        # ir = cv.equalizeHist(ir)
        ir = cv.cvtColor(ir, cv.COLOR_GRAY2BGR)

        # ir2[mask==0]=(0)
        # ir2 = histstretch(ir2)
        # log(f"{ir2.shape=} {ir2.dtype=} {ir2.max()=} {ir2.min()=}")
        ir2 = cv.normalize(ir2, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
        # ir2 = cv.equalizeHist(ir2)
        ircolor = cv.cvtColor(ir2, cv.COLOR_GRAY2BGR)
        frame = ircolor.copy()

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

    # rgba = c.transformed_color
    rgba = None
    if rgba is not None:
        rgb = cv.cvtColor(rgba, cv.COLOR_BGRA2BGR)
        rgb2 = rgb.copy()
        rgb2[mask == 0] = (0, 0, 0)
        ishow("rgb2", rgb2)

    if True:
        depth = depth_raw.copy()
        #depth[depth>4000] = 0
        depth = cv.convertScaleAbs(depth, None, 1/8)
        #depth = cv.normalize(depth_raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        depth = cv.applyColorMap(depth, cv.COLORMAP_JET)
        log(f"{depth_raw.min()=} {depth_raw.max()=}")
        ishow("depth", depth)
    if color is not None:
        ishow("color", color)
    ishow("ir", ir)
    ishow("ir2", ir2)
    ishow("ircolor", ircolor)
    # ishow("depth",depth)

    return frame
