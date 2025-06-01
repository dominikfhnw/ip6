import numpy as np
import cv2 as cv
from isave import ishow, save_data
import log
import meta
if meta.true("kinect_enable"):
    import kinect_raw
else:
    import kinect_fake as kinect_raw
import timey
from imagefunctions import avg

log, dbg, logger = log.auto(__name__)

depth_stab = []
ir_stab = []
AVG_Depth = False
AVG_IR = False

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
    # keep grey small if we don't prescale
    grey = np.full((ROI_SCALED, ROI_SCALED, 3), 127, np.dtype('uint8'))
    # output image is always size*scale
    return ROI_SIZE*ROI_SCALE, ROI_SIZE*ROI_SCALE


def get():
    return kinect_raw.get()


def roi(img):
    roi = np.array(img[ROI_Y:ROI_Y+ROI_SIZE, ROI_X:ROI_X+ROI_SIZE])
    roi = np.rot90(roi, axes=(0,1))
    if ROI_PRESCALE and ROI_SCALE != 1:
        roi = cv.resize(roi, None, None, ROI_SCALE, ROI_SCALE, ROI_ALG)
    return roi

def roi_color(img,scaler):
    roi = np.array(img[int(ROI_Y*scaler):int((ROI_Y+ROI_SIZE)*scaler), int(ROI_X*scaler):int((ROI_X+ROI_SIZE)*scaler)])
    roi = np.rot90(roi, axes=(0,1))
    return roi

def scale(img, alg=ROI_ALG):
    if ROI_PRESCALE or ROI_SCALE == 1:
        return img
    else:
        return cv.resize(img, None, None, ROI_SCALE, ROI_SCALE, alg)

# Select distance range
def range_mask(depth):
    lo = meta.num("kinect_lo")
    if lo < 1:
        lo = 1
    lo = np.array([lo])
    hi = np.array([meta.num("kinect_hi")])
    mask = cv.inRange(depth, lo, hi)
    if meta.true("kinect_fast"):
        mask2 = None
    else:
        mask2 = np.array(mask, dtype=bool)
        mask2 = np.stack((mask2,)*3, axis=-1)
    return mask, mask2

def depth_relative(depth, mask, bool_mask, name="dr2"):
    dr2 = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U, mask=mask)
    dr2 = cv.applyColorMap(dr2, cv.COLORMAP_JET)
    # This uses more CPU power than it should...
    dr2 = np.where(bool_mask, dr2, grey)
    #dr2[bool_mask] = np.array([127,127,127])
    #dr2[mask==0]=np.array([127,127,127])
    dr2 = scale(dr2)
    ishow(name, dr2, True)
    return dr2


def depth_relative2(depth, mask, bool_mask, name="dr2"):
    depth[mask==0]=0
    maxd = depth.max()      # find the deepest point in the *masked* area
    depth[depth==0]=maxd    # set stuff to that
    depth[mask==0]=maxd

    gg = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    gg = 255-gg
    #gg = (0.7*gg).astype('u1')
    gg = scale(gg, cv.INTER_NEAREST)
    ishow("gg", gg)
    return gg


def depth_absolute(depth):
    #depth = depth_raw.copy()
    #depth[depth>4000] = 0
    max=3000    # max distance in mm
    depth = cv.convertScaleAbs(depth, None, 255/max)
    #depth = cv.normalize(depth_raw, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    depth = cv.applyColorMap(depth, cv.COLORMAP_JET)
    #log(f"{depth_raw.min()=} {depth_raw.max()=}")
    ishow("depth", depth)

def process_ir_full(ir, name="ir"):
    ir = cv.normalize(ir, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    ir = cv.equalizeHist(ir)
    if AVG_IR:
        global ir_stab
        ir_stab.append(ir)
    cv.rectangle(ir, (ROI_X,ROI_Y),(ROI_X+ROI_SIZE,ROI_Y+ROI_SIZE), (255,255,255), 2)
    ishow(name, ir)


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
    ircolor = cv.cvtColor(ircolor, cv.COLOR_GRAY2BGR)
    ircolor[depth_raw == maxVal]=(0,0,255)
    ircolor[depth_raw == minVal]=(255,0,0)

    cv.circle(ircolor, center=minLoc, radius=3, color=(255, 0, 0), thickness=-1)
    cv.circle(ircolor, center=maxLoc, radius=3, color=(0, 0, 255), thickness=-1)
    # cv.line(anno, pt1=index, pt2=base, thickness=2, color=(0, 165, 255))
    cv.putText(ircolor, f"{minVal / 10: .1f}cm",
               (minLoc[0] + 10, minLoc[1] + 10), cv.FONT_HERSHEY_SIMPLEX,
               0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(ircolor, f"{maxVal / 10: .1f}cm",
               (maxLoc[0] + 10, maxLoc[1] + 10), cv.FONT_HERSHEY_SIMPLEX,
               0.5, (0, 255, 0), 1, cv.LINE_AA)
    ircolor = scale(ircolor)
    ishow("ircolor", ircolor)


def process_rgb(rgba, mask, ir):
    rgb = cv.cvtColor(rgba, cv.COLOR_BGRA2BGR)
    rgb2 = rgb.copy()
    rgb2[mask == 0] = (0, 0, 0)
    rgb = scale(rgb)
    rgb2 = scale(rgb2)
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
    depth_raw, ir, rgba, orig_color = kinect_raw.get()
    if depth_raw is None:
        return np.zeros((meta.num("height"), meta.num("width"), 3), np.dtype('u1'))

    depth_orig=depth_raw.copy()
    depth_raw[depth_raw==0]=65535
    roi_depth=roi(depth_raw)
    roi_mask, roi_bool_mask = range_mask(roi_depth)
    if AVG_Depth:
        global depth_stab
        depth_stab.append(roi_depth)
        avg_depth = avg(depth_stab,500, 'float32', 'float32')
        save_data("avg_depth", avg_depth=avg_depth, depth_raw=depth_raw)
        avg_roi_mask, avg_roi_bool_mask = range_mask(avg_depth.astype(np.dtype('uint16')))
        depth_relative(avg_depth, avg_roi_mask, avg_roi_bool_mask, "dr avg")

    if meta.true("kinect_fast"):
        frame = process_ir(roi(ir),roi_mask)
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        timey.delta(__name__, t1)
        return frame

    mask, bool_mask = range_mask(depth_raw)

    #log(f"{mask.shape=} {mask.dtype=} {bool_mask.shape=} {bool_mask.dtype=}")
    if True:
        depth_relative(roi_depth, roi_mask, roi_bool_mask)
        f = depth_relative2(roi(depth_orig), roi_mask, roi_bool_mask)
        frame = cv.cvtColor(f, cv.COLOR_GRAY2BGR)

    if ir is not None:
        proc = process_ir(roi(ir),roi_mask)
        #log(f"{frame.shape=} {frame.dtype=}")
        if not meta.true("kinect_fast"):
            process_ir_full(ir)
            if AVG_IR:
                process_ir_full(avg(ir_stab,10), "ir stab")
            #process_ir_debug(roi(ir), roi_depth, roi_mask)
        proc = scale(proc)
        #frame = cv.cvtColor(proc, cv.COLOR_GRAY2BGR)
    else:
        log("IR failed")
        exit(12)

    if rgba is not None:
        process_rgb(roi(rgba), roi_mask, roi(ir))
        if orig_color is not None:
            co2 = np.array(orig_color[:, 171:1878])
            color_scale = 1536 / 576
            cv.rectangle(co2, (int(ROI_X*color_scale),int(ROI_Y*color_scale)),(int((ROI_X+ROI_SIZE)*color_scale),int((ROI_Y+ROI_SIZE)*color_scale)), (255,255,255), 2)
            frame = roi_color(co2, color_scale)
            frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

    if meta.true("kinect_depth"):
        depth_absolute(depth_raw)
    timey.delta(__name__, t1)
    return frame
