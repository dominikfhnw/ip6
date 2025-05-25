import cv2 as cv
import log
import meta

Enabled = False

log, dbg, logger = log.auto(__name__)
def isave(img, name="out", force=False):
    if not force and not Enabled:
        return
    fname = name.replace(" ", "_")
    frame = meta.get('frame')
    start = meta.get('start_timestr')
    var = f"{start}-{frame}"
    filename = "out/"+fname+var+".jpg"
    filename = f"gesture/{frame}-{start}-{fname}.jpg"
    if cv.imwrite(filename, img):
        log("Written "+filename)
    else:
        logger.fatal("Error writing file "+filename)
        exit()

def ishow(name, img, save:bool=False):
    if save and meta.true('save'):
        isave(img, name, force=True)

    if meta.true('gui'):
        cv.imshow(name, img)

