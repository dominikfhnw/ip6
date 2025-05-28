import pyk4a
from pyk4a import Config, PyK4A
import meta
import numpy as np
import log

log, dbg, logger = log.auto(__name__)

Wide = meta.true("kinect_wide")
Color = meta.true("kinect_color")
width, height = None, None

def init():
    dbg("start init")
    if Color:
        res = pyk4a.ColorResolution.RES_1536P
    else:
        res = pyk4a.ColorResolution.OFF

    global width, height
    if not Wide:
        mode = pyk4a.DepthMode.NFOV_UNBINNED
        width = 640
        height = 576
    else:
        mode = pyk4a.DepthMode.WFOV_2X2BINNED
        width = 512
        height = 512

    global k4a
    k4a = PyK4A(
        Config(
            color_resolution=res,
            depth_mode=mode,
            camera_fps=pyk4a.FPS.FPS_30,
            synchronized_images_only=False,
        )
    )

    meta.set("width", width)
    meta.set("height",height)

    k4a.start()
    dbg("end init")
    return width, height

def get():
    try:
        cap = k4a.get_capture(50)
    except:
        log(f"capture timeout")
        return None, None, None

    if Color:
        color = cap.transformed_color
        if color is None:
            color = np.zeros((height, width, 4), np.dtype('u1'))
            log("borked color")
        return cap.depth, cap.ir, color
    return cap.depth, cap.ir, None
