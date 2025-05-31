import numpy as np
import cv2 as cv
from isave import ishow
import log
import meta
if meta.true("kinect_enable"):
    import kinect_raw
else:
    import kinect_fake as kinect_raw
import timey

log, dbg, logger = log.auto(__name__)

ROI_X=210
ROI_Y=10
ROI_SIZE=300
ROI_ROTATE=cv.ROTATE_90_COUNTERCLOCKWISE
ROI_SCALE=2
ROI_PRESCALE=False
if ROI_PRESCALE:
    ROI_SCALED=ROI_SIZE*ROI_SCALE
else:
    ROI_SCALED=ROI_SIZE
ROI_ALG=cv.INTER_LINEAR

def init():
    width, height = kinect_raw.init()
    global grey
    grey = np.full((ROI_SCALED, ROI_SCALED, 3), 127, np.dtype('uint8'))
    return width, height


def get():
    return kinect_raw.get()


def roi(img):
    roi = np.array(img[ROI_Y:ROI_Y+ROI_SIZE, ROI_X:ROI_X+ROI_SIZE])
    roi = np.rot90(roi, axes=(0,1))
    if ROI_PRESCALE and ROI_SCALE != 1:
        roi = cv.resize(roi, None, None, ROI_SCALE, ROI_SCALE, ROI_ALG)
    return roi


def scale(img):
    if ROI_PRESCALE or ROI_SCALE == 1:
        return img
    else:
        return cv.resize(img, None, None, ROI_SCALE, ROI_SCALE, ROI_ALG)

# Select distance range
def range_mask(depth):
    lo = np.array([meta.num("kinect_lo")])
    hi = np.array([meta.num("kinect_hi")])
    mask = cv.inRange(depth, lo, hi)
    if meta.true("kinect_fast"):
        mask2 = None
    else:
        mask2 = np.array(mask, dtype=bool)
        mask2 = np.stack((mask2,)*3, axis=-1)
    return mask, mask2

def depth_relative(depth, mask, bool_mask):
    dr2 = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
    dr2 = cv.applyColorMap(dr2, cv.COLORMAP_JET)
    # This uses more CPU power than it should...
    dr2 = np.where(bool_mask, dr2, grey)
    #dr2[bool_mask] = np.array([127,127,127])
    #dr2[mask==0]=np.array([127,127,127])
    dr2 = scale(dr2)
    ishow("dr2", dr2, True)

def depth_absolute(depth):
    #depth = depth_raw.copy()
    #depth[depth>4000] = 0
    max=3000    # max distance in mm
    depth = cv.convertScaleAbs(depth, None, 255/max)
    #depth = cv.normalize(depth_raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    depth = cv.applyColorMap(depth, cv.COLORMAP_JET)
    #log(f"{depth_raw.min()=} {depth_raw.max()=}")
    ishow("depth", depth)

def process_ir_full(ir):
    ir = cv.normalize(ir, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    ir = cv.equalizeHist(ir)
    cv.rectangle(ir, (ROI_X,ROI_Y),(ROI_X+ROI_SIZE,ROI_Y+ROI_SIZE), (255,255,255), 2)
    ishow("ir", ir)


def process_ir_mask(ir, mask):
    ir = cv.normalize(ir, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
    return ir


def process_ir(ir, mask):
    # fix the ultra bright "too much reflection" pixels
    imax = ir.max() + 1
    ir[ir == 65535] = imax

    return process_ir_mask(ir, mask)


def process_ir_debug(img, depth_raw, mask):
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(depth_raw, mask)
    ircolor = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
    ircolor = cv.equalizeHist(ircolor)
    #ircolor[depth_raw == maxVal]=(0,0,255)
    #ircolor[depth_raw == minVal]=(255,0,0)

    cv.circle(ircolor, center=minLoc, radius=3, color=(255, 0, 0), thickness=-1)
    cv.circle(ircolor, center=maxLoc, radius=3, color=(0, 0, 255), thickness=-1)
    # cv.line(anno, pt1=index, pt2=base, thickness=2, color=(0, 165, 255))
    cv.putText(ircolor, f"{minVal / 10: .1f}cm",
               (minLoc[0] + 10, minLoc[1]), cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 255, 255), 1, cv.LINE_AA)
    cv.putText(ircolor, f"{maxVal / 10: .1f}cm",
               (maxLoc[0] + 10, maxLoc[1]), cv.FONT_HERSHEY_SIMPLEX,
               1, (0, 255, 255), 1, cv.LINE_AA)
    ishow("ircolor", ircolor)


def process_rgb(rgba, mask, ir):
    rgb = cv.cvtColor(rgba, cv.COLOR_BGRA2BGR)
    rgb2 = rgb.copy()
    rgb2[mask == 0] = (0, 0, 0)
    ishow("rgb2", rgb2, True)
    ishow("color", rgb)

    if meta.true("kinect_composite"):
        rgb3 = rgb.copy()
        black = np.array([0,0,0])
        mask2 = cv.inRange(rgb3, black, black)
        locs = np.where(mask2 != 0)
        ishow("mask2",mask2)
        img1 = cv.normalize(ir, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        img1 = cv.equalizeHist(img1)
        img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        ishow("img1",img1)
        rgb3[locs[0], locs[1]] = img1[locs[0], locs[1]]
        ishow("composite", rgb3, True)


def process():
    t1 = timey.time()
    depth_raw, ir, rgba = kinect_raw.get()
    if depth_raw is None:
        return np.zeros((meta.num("height"), meta.num("width"), 3), np.dtype('u1'))

    roi_depth=roi(depth_raw)
    roi_mask, roi_bool_mask = range_mask(roi(depth_raw))
    mask, bool_mask = range_mask(depth_raw)

    if meta.true("kinect_fast"):
        frame = process_ir(ir,mask)
        timey.delta(__name__, t1)
        return frame

    #log(f"{mask.shape=} {mask.dtype=} {bool_mask.shape=} {bool_mask.dtype=}")
    if True:
        depth_relative(roi_depth, roi_mask, roi_bool_mask)

    if ir is not None:
        proc = process_ir(roi(ir),roi_mask)
        #log(f"{frame.shape=} {frame.dtype=}")
        if not meta.true("kinect_fast"):
            process_ir_full(ir)
            #process_ir_debug(ir, depth_raw, mask)
        proc = scale(proc)
        frame = cv.cvtColor(proc, cv.COLOR_GRAY2BGR)
    else:
        log("IR failed")
        exit(12)

    if rgba is not None:
        process_rgb(rgba, mask, ir)

    if meta.true("kinect_depth"):
        depth_absolute(depth_raw)
    timey.delta(__name__, t1)
    return frame
