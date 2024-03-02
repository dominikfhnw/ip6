import log
import cv2 as cv
import meta
import numpy as np

log, dbg, logger = log.auto(__name__)

#FACTOR = 3

def process(img, height):
    # TODO: ugly
    global FACTOR
    FACTOR=int(height/100)

    img = img - np.min(img)
    img = img * (256/np.max(img))
    img = cv.cvtColor(img.astype("uint8"), cv.COLOR_GRAY2BGR)

    x1, y1 = 0,0
    start=0
    l = np.empty((6,7))

    xoffset, xmax, xstep = 12, 250, 46
    for i in range(xoffset, xmax, xstep):
        x = letter(img, i,x1,y1,str(start))
        l[start] = x
        start += 1
    dbg("NUMBER: "+str(l))
    min = np.min(l)
    #l2 = np.subtract(l, min)
    l2 = l - min
    max = np.max(l2)
    l2 = l2 / max
    l2 = 1 - l2
    l2 = np.round(l2)
    dbg("NUMBER2: "+str(l2))

    # alternative font
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
    i=0
    out=""
    for num in l2:
        result=-1
        dbg(f"NUM {i}: {l2[i]}")
        j=0
        for ref in digits:
            if np.array_equal(digits[j], l2[i]):
                dbg(f"MATCH {j}")
                result=j
            j += 1
        if result >= 0:
            n = str(result)
        else:
            n = "#"
        out += n
        cv.putText(img, n,
                   (FACTOR*(xoffset+i*xstep),FACTOR*95),
                   fontFace=cv.FONT_HERSHEY_DUPLEX,
                   fontScale=1,
                   color=(255, 0, 0)
        )
        i += 1
    cv.imshow("ocr", img)
    log(f"Number: {out}")

def rec(img,x,y,w,h,color=(0,0,255), name=None):
    x2=x+w
    y2=y+h

    x *= FACTOR
    y *= FACTOR
    x2 *= FACTOR
    y2 *= FACTOR

    #log("SHAPE "+str(img.ndim))
    if img.ndim == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    i2 = gray[y:y2, x:x2]
    #cv.imshow("seg", i2)
    #cv.waitKey(0)
    #log("SHAPE "+str(i2.shape))

    pixsum = int(cv.sumElems(i2)[0])
    #log("SUM "+name+" "+str(pixsum))
    #log(f"SEG {name} ({x},{y}) ({x2},{y2})")
    cv.rectangle(img, (x,y), (x2,y2), color)
    return pixsum

def letter(img,x,x1,y1,name=None):
    y=11
    green=(0,255,0)
    s = dict()
    rec(img,x,y,36,73, name=name)
    s["a"] = rec(img,x+8,y+2,20,5, green, name+":a")
    s["g"] = rec(img,x+8,y+33,20,5, green, name+":g")
    s["d"] = rec(img,x+8,y+64,20,5, green, name+":d")

    s["f"] = rec(img,x+2,y+10,5,20, green, name+":f")
    s["b"] = rec(img,x+29,y+10,5,20, green, name+":b")
    s["e"] = rec(img,x+2,y+41,5,20, green, name+":e")
    s["c"] = rec(img,x+29,y+41,5,20, green, name+":c")

    t = np.array([s["a"], s["b"], s["c"], s["d"], s["e"], s["f"], s["g"]])


    #log("SEGMENTS: "+str(s))
    return t

def seg():
    file = "roi-thresh20240229-220608.jpg"
    #file = "composite20240301-160253.jpg"
    file = "roi-thresh20240229-222536.jpg"
    file = "roi-thresh20240301-145649.jpg"
    file ="composite20240301-165201.jpg"

    file = "composite20240301-160253.jpg" # known good
    file="roi-gray20240302-212209.jpg"
    input = cv.imread("out/"+file)
    cv.imshow("seg", input)
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
