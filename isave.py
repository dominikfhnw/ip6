import cv2 as cv
from timestring import timestring
import log
import meta

Enabled = False

log, dbg, logger = log.auto(__name__)
def isave(img, name="out", force=False):
    if not force and not Enabled:
        return
    filename = "out/"+name+timestring()+".jpg"
    if cv.imwrite(filename, img):
        log("Written "+filename)
    else:
        logger.fatal("Error writing file "+filename)
        exit()

def ishow(name, img):
    if meta.true('save'):
        isave(img, name, force=True)

    cv.imshow(name, img)

