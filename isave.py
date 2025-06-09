import cv2 as cv
import numpy as np
import log
import meta

Enabled = False

log, dbg, logger = log.auto(__name__)

def fname(name):
    fname = name.replace(" ", "_")
    frame = meta.get('frame')
    start = meta.get('start_timestr')
    var = f"{start}-{frame}"
    filename = "gesture/"+fname+var
    #filename = f"gesture/{frame}-{start}-{fname}"
    return filename

def isave(img, name="out", force=False):
    if not force and not Enabled:
        return
    filename = fname(name)+".jpg"
    if cv.imwrite(filename, img):
        log("Written "+filename)
    else:
        logger.fatal("Error writing file "+filename)
        exit()

def save_data(name,force:bool=False, *args, **kwds):
    if not force and not meta.true('save'):
        return
    filename = fname(name)
    np.savez_compressed(filename, meta=meta.meta, *args, **kwds)
    log("Written data "+filename+".npz")

def ishow(name, img:np.ndarray, save:bool=False):
    if img is None:
        log(f"ishow None {name=}")
        return
    if save and meta.true('save'):
        isave(img, name, force=True)

    if meta.true('gui'):
        cv.imshow(name, img)

