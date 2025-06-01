import pyk4a
from pyk4a import Config, PyK4A, PyK4ARecord, PyK4APlayback
import meta
import numpy as np
import log
from isave import save_data, fname
import timey
import math
import atexit

log, dbg, logger = log.auto(__name__)

Wide = meta.true("kinect_wide")
Color = meta.true("kinect_color")
Unbinned = meta.true("kinect_wide_unbinned")
Passive = meta.true("kinect_passive")
Record = meta.true("kinect_record")
Playback = meta.true("kinect_playback")
width, height = None, None
record = None

def init():
    dbg("start init")
    global width, height
    if Playback:
        fake = PyK4A()
        fake.open()
        global playback
        playback = PyK4APlayback(meta.get("kinect_file"))
        playback.open()
        playback._calibration = fake.calibration
        log(f"Record length: {playback.length / 1000000: 0.2f} sec")
        capture = playback.get_next_capture()
        height, width = capture.depth.shape
        log(f"{capture.depth.shape=}")
        playback.seek(0)
        meta.set("width", width)
        meta.set("height",height)
        return width, height
    if Color:
        res = pyk4a.ColorResolution.RES_1536P
        sync = True
    else:
        res = pyk4a.ColorResolution.OFF
        sync = False

    fps = pyk4a.FPS.FPS_30
    if Passive:
        mode = pyk4a.DepthMode.PASSIVE_IR
        width = 1024
        height = 1024
        fps = pyk4a.FPS.FPS_30
    elif not Wide:
        mode = pyk4a.DepthMode.NFOV_UNBINNED
        width = 640
        height = 576
    else:
        if Unbinned:
            mode = pyk4a.DepthMode.WFOV_UNBINNED
            width = 1024
            height = 1024
            fps = pyk4a.FPS.FPS_15
        else:
            mode = pyk4a.DepthMode.WFOV_2X2BINNED
            width = 512
            height = 512

    global k4a
    k4a = PyK4A(
        Config(
            color_resolution=res,
            depth_mode=mode,
            camera_fps=fps,
            synchronized_images_only=sync,
        )
    )

    meta.set("width", width)
    meta.set("height",height)

    try:
        k4a.start()
        if Record:
            global record
            record = PyK4ARecord(config=k4a._config, path=fname("kinect")+".k4a")
            record.create()
    except Exception as ex:
        logger.fatal(f"Opening Kinect failed: {ex=}")
        exit(3)
    dbg("end init")
    return width, height

def get():
    t1 = timey.time()
    try:
        if Playback:
            cap = playback.get_next_capture()
            return cap.depth, cap.ir, cap.color, None
        cap = k4a.get_capture(50)
        if Record:
            record.write_capture(cap)
        imu = k4a.get_imu_sample(50)
    except EOFError:
        log("end of stream")
        exit(0)
    except pyk4a.errors.K4ATimeoutException:
        log(f"capture timeout")
        return None, None, None, None
    except Exception as ex:
        log(f"exception: {ex=} {ex.args=} {type(ex)=}")
        exit(5)

    x,y,z = imu["acc_sample"]
    temperature = imu["temperature"]
    tilt = -(z/9.81)
    angle = 90-math.degrees(math.acos(tilt))
    dbg(f"{angle=:.2f}° {temperature=:.1f}°C")

    color = None
    color2 = None
    if Passive:
        depth = np.zeros((height, width), np.dtype('u2'))
        return depth, cap.ir, None, None
    if Color:
        color = cap.transformed_color
        color2 = cap.color
        if color is None:
            color = np.zeros((height, width, 4), np.dtype('u1'))
            log("borked color")
    depth, ir = cap.depth, cap.ir
    save_data("kinect", depth=depth, ir=ir, imu=imu, color=color)
    timey.delta(__name__,t1)
    return depth, ir, color, color2


def _end():
    if record is not None:
        log("finalizing record")
        record.flush()
        record.close()
        log(f"{record.path}: {record.captures_count} frames written.")


atexit.register(_end)