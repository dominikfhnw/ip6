import numpy as np
import cv2 as cv

__all__ = ['histstretch', 'avg', 'phaseCorrelate', 'warpAffine', 'translationMatrix', 'stabilize', 'otsu', 'adaptivethresh', 'otsu_linearize']

OTSU_SCALE = 1

# TODO: only works with uint8
def histstretch(img):
    img = img - np.min(img)
    return (img * (255/np.max(img))).astype("uint8")

# TODO: use numpy for speed
# TODO: image array also grows without bound
def avg(image, count):
    assert len(image) < 2**(32-8)
    composite = np.zeros(shape=image[-1].shape, dtype=np.uint32)
    last = image[-count:]
    for i in last:
        composite += i
    return (composite/count).astype('uint8')

def phaseCorrelate(img1, img2):
    #log("SHAPE:"+str(img2.shape))
    if img1.ndim == 3:
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    if img2.ndim == 3:
        img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret = cv.phaseCorrelate(np.float32(img1),np.float32(img2))
    #log("correlation signal power: "+str(ret[1]))
    return ret[0]

def warpAffine(img, M, borderValue=(255,255,255)):
    return cv.warpAffine(img, M, (img.shape[1], img.shape[0]),borderValue=borderValue)

def translationMatrix(point):
    return np.array([
        [1, 0, point[0]],
        [0, 1, point[1]]
    ]).astype('float32')

def stabilize(img, reference):
    r = phaseCorrelate(img, reference)
    M = translationMatrix(r)
    return warpAffine(img, M)

def otsu(img):
    if OTSU_SCALE == 1:
        return cv.threshold(img,0,1000,cv.THRESH_BINARY+cv.THRESH_OTSU)
    interpolation = cv.INTER_CUBIC
    img = cv.resize(img, None, None, OTSU_SCALE, OTSU_SCALE, interpolation)
    ret, img = cv.threshold(img,None,None,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #log(img.shape)
    return ret, cv.resize(img, None, 1000, 1/OTSU_SCALE, 1/OTSU_SCALE, interpolation)

def otsu_map(input, thresh):
    norm = thresh/127
    if input <= thresh:
        r = input / norm
        return r
    else:
        r = input - thresh
        max = 255 - thresh
        scale = 255 / max
        r = r * scale
        r = 127 + r/2

        return r

def otsu_linearize2(img):
    thresh_full, _ = otsu(img)
    thresh = thresh_full/255
    #dbg(f"LLIIIIIII {thresh_full} {thresh}")

    #return np.where(img < thresh, int(img * thresh), int((255-img)*thresh))
    map = np.vectorize(otsu_map, otypes=[img.dtype])
    return map(img, thresh_full)

def otsu_linearize(img):
    thresh, _ = otsu(img)
    norm = 2*thresh/255
    scale = 255/(255-thresh)
    return np.where(img < thresh, img / norm, (255+(img - thresh) * scale)/2).astype(np.uint8)

def adaptivethresh(img):
    return cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 39, 4)

