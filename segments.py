import log
import cv2 as cv
import meta
import numpy as np
from imagefunctions import *

log, dbg, logger = log.auto(__name__)

def process(img, height):
    # TODO: ugly
    # Global scaling factor. Initially all input images were 100px high
    # This keeps the same constants for all possible input image heights,
    # and allows the constants to be treated as "from 100%"
    global FACTOR
    FACTOR=int(height/100)

    img = histstretch( img )
    thresh, ots = otsu(img)
    thresh_norm = thresh/255
    log(f"THRESHOLD: {thresh}")
    #_, img = cv.threshold(img,thresh-20,1000,cv.THRESH_BINARY)
    if meta.true("thresh"):
        gauss = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 39, 4)
        img = gauss

    gui = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    xoffset, xmax, xstep = 12, 250, 46
    mat = extractdigits(img, gui, xoffset, xmax, xstep)
    dbg("MAT1 "+str(mat))
    mat = normalize_matrix(mat)
    mat = threshold_matrix(mat, thresh_norm)
    dbg("MAT2 "+str(mat))

    digits = []
    for i, digit in enumerate(mat):
        dbg(f"NUM {i}: {mat[i]}")
        digits.append(digit_match(digit))

    for i, n in enumerate(digits):
        text(gui, n, xoffset+i*xstep, 95)
    cv.imshow("ocr", gui)

    out = ''.join(digits)
    meta.set("result", out)
    log(f"Number: {out}")

def text(gui, text, x, y):
    cv.putText(gui, text,
                (FACTOR*x,FACTOR*y),
               fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=1,
               color=(255, 0, 0)
               )


def extractdigits(img, gui, xoffset, xmax, xstep):
    start=0
    l = np.empty((6,7))

    for i in range(xoffset, xmax, xstep):
        x = extractdigit(img, gui, i)
        l[start] = x
        start += 1
    dbg("NUMBER: "+str(l))

    return l

# normalize to range [0,1]
def normalize_matrix(l):
    min = np.min(l)
    l2 = l - min
    max = np.max(l2)
    l2 = l2 / max

    return l2

def threshold_matrix(l, thresh):
    if meta.get("invertDigits"): # invert
        l = 1 - l
        thresh = 1 - thresh
    # Use Otsu's value as threshold
    l = (l > thresh).astype("int");
    return l

def digit_match(digit, font=0):
    # alternative font. single character to not mess up the array below
    a = 0
    digits = np.array([
        [1, 1, 1, 1, 1, 1, 0], # 0
        [0, 1, 1, 0, 0, 0, 0], # 1
        [1, 1, 0, 1, 1, 0, 1], # 2
        [1, 1, 1, 1, 0, 0, 1], # 3
        [0, 1, 1, 0, 0, 1, 1], # 4
        [1, 0, 1, 1, 0, 1, 1], # 5
        [a, 0, 1, 1, 1, 1, 1], # 6
        [1, 1, 1, 0, 0, 0, 0], # 7
        [1, 1, 1, 1, 1, 1, 1], # 8
        [1, 1, 1, a, 0, 1, 1], # 9
        #[0, 0, 0, 0, 0, 0, 1], # -
        #[0, 0, 0, 0, 0, 0, 0], # (blank)
    ])
    result = '#'
    j=0
    for ref in digits:
        if np.array_equal(ref, digit):
            dbg(f"MATCH {j}")
            result=str(j)
        j += 1

    return result


def rec(img,gui,x,y,w,h,color=(0,255,0)):
    x2=x+w
    y2=y+h

    x *= FACTOR
    y *= FACTOR
    x2 *= FACTOR
    y2 *= FACTOR

    i2 = img[y:y2, x:x2]

    pixsum = int(cv.sumElems(i2)[0])

    cv.rectangle(gui, (x,y), (x2,y2), color)
    return pixsum

def extractdigit(img, gui, x):
    y=11
    s = dict()

    s["a"] = rec(img,gui,x+8,y+2,20,5)
    s["g"] = rec(img,gui,x+8,y+33,20,5)
    s["d"] = rec(img,gui,x+8,y+64,20,5)

    s["f"] = rec(img,gui,x+2,y+10,5,20)
    s["b"] = rec(img,gui,x+29,y+10,5,20)
    s["e"] = rec(img,gui,x+2,y+41,5,20)
    s["c"] = rec(img,gui,x+29,y+41,5,20)

    t = np.array([s["a"], s["b"], s["c"], s["d"], s["e"], s["f"], s["g"]])

    return t

def seg():
    file = "roi-thresh20240229-220608.jpg"
    #file = "composite20240301-160253.jpg"
    file = "roi-thresh20240229-222536.jpg"
    file = "roi-thresh20240301-145649.jpg"
    file ="composite20240301-165201.jpg"

    file = "composite20240301-160253.jpg" # known good
    #file="roi-gray20240302-212209.jpg"
    #file="needs-local-thresh.png"
    #file="smeared.png"
    input = cv.imread("out/"+file)
    cv.imshow("seg", input)
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    #input = histstretch( input )

    #segments.process(input, None)
    #

    while True:
        img = input.copy()
        log(str(img.shape[0]))
        process(img, img.shape[0])
        #dbg(f'MIN {min} MAX {max}')
        cv.imshow("seg", img)
        cv.waitKey(0)
        exit()

        match chr(cv.waitKey(int(1e9)) & 0xFF).lower():
            case 'q':
                break
            case 'w':
                y1 -= 1
            case 's':
                y1 += 1
            case 'a':
                x1 -= 1
            case 'd':
                x1 += 1
        #if closed("seg"):
        #    break
        log(f'x:{x1}, y:{y1}')
    exit()
