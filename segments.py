import log
from imagefunctions import *
import match

log, dbg, logger = log.auto(__name__)

def process(img, height):
    # TODO: ugly
    # Global scaling factor. Initially all input images were 100px high
    # This keeps the same constants for all possible input image heights,
    # and allows the constants to be treated as "from 100%"
    global FACTOR
    FACTOR=int(height/100)
    dbg(f"FACTOR: {FACTOR}")

    img = histstretch( img )
    thresh, ots = otsu(img)
    thresh_norm = thresh/255
    log(f"THRESHOLD: {thresh}")
    #_, img = cv.threshold(img,thresh-20,1000,cv.THRESH_BINARY)
    gauss = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 39, 4)
    #gauss2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 4)

    # if meta.true("thresh"):
    #     gauss = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 39, 4)
    #     img = gauss

    guiocr(img, thresh_norm)
    guiocr(gauss, thresh_norm, "ocr2")
    #guiocr(gauss2, thresh_norm, "ocr2b")
    #guiocr(gauss, thresh_norm, "ocr3", factor=3)
    #guiocr(gauss2, thresh_norm, "ocr3b", factor=3)

def guiocr(img, threshold=0.5, name="ocr", factor:int=1):
    # TODO: get via meta? or as param from lower layer
    xoffset, xmax, xstep = 12, 250, 46

    global FACTOR
    if factor != 1:
        dbg(f"factor before: {FACTOR}")
        FACTOR = int(FACTOR/factor)
        dbg(f"factor after: {FACTOR}")
        img = cv.resize(img, None, None, 1/factor, 1/factor, cv.INTER_CUBIC)

    gui = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    mat = extractdigits(img, gui, xoffset, xmax, xstep)
    digits = match.process(mat, threshold)
    if factor != 1:
        FACTOR = int(FACTOR*factor)
        #img = cv.resize(img, None, None, factor, factor, cv.INTER_CUBIC)
        gui = cv.resize(gui, None, None, factor, factor, cv.INTER_CUBIC)

    for i, tuple in enumerate(digits):
        n, conf = tuple
        text(gui, n, xoffset+i*xstep, 95)
        text(gui, str(conf), xoffset+i*xstep+15, 95, (0,0,255))

    cv.imshow(name, gui)

def text(gui, text, x, y, color=(255,0,0)):
    cv.putText(gui, text,
                (FACTOR*x,FACTOR*y),
               fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=FACTOR/3,
               color=color
               )


def extractdigits(img, gui, xoffset, xmax, xstep):
    start=0
    l = np.empty((6,7))

    for i in range(xoffset, xmax, xstep):
        x = extractdigit(img, gui, i)
        l[start] = x
        start += 1

    return l


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
    file = "out/composite20240301-160253.jpg"
    #file = "out/roi-thresh20240229-222536.jpg"
    #file = "out/roi-thresh20240301-145649.jpg"
    #file ="out/composite20240301-165201.jpg"

    #file = "out/composite20240301-160253.jpg" # known good
    #file="out/roi-gray20240302-212209.jpg"
    #file="needs-local-thresh.png"
    #file="smeared.png"
    file="aruco-webcam.png"
    file="out/roi-gray20240229-191400.jpg"
    file="out/detect-roi20240229-150920.jpg"
    file="out/roi-gray20240301-143853.jpg"
    file="needs-local-thresh.png"
    #file = "out/composite20240301-160253.jpg" # known good

    input = cv.imread(""+file)
    cv.imshow("seg", input)
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    #input = histstretch( input )
    #input = cv.resize(input, None, None, 3, 3, cv.INTER_NEAREST)
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
