import pyk4a
from pyk4a import Config, PyK4A
import meta

Wide = meta.true("kinect_wide")
Color = meta.true("kinect_color")

def init():
    if Color:
        res = pyk4a.ColorResolution.RES_1536P
    else:
        res = pyk4a.ColorResolution.OFF

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
    return width, height

def get():
    cap = k4a.get_capture();
    return cap.depth, cap.ir
