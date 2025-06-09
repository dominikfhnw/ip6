import log
import meta
import numpy as np
from imagefunctions import p

log, dbg, logger, isdbg = log.auto2(__name__)

def process(mat, threshold, binary=False, name=None, cutoff=0):
    dbg("started match")

    digits1 = []
    digits2 = []

    str1, str2 = '', ''
    err1, err2 = 0,0
    for i, digit in enumerate(mat):
        dbg(f"NUM: {i}")
        result1, err  = thresh_match(digit, threshold, binary=binary)
        result2, rmsd = mse_match(digit, binary=binary)

        str2 += f"[{result2}] {p(rmsd)} "
        str1 += f"[{result1}] {p(err)} "
        digits1.append((result1, err))
        digits2.append((result2,rmsd))
        err1 += err**2
        err2 += rmsd**2
        dbg(f"ERRADD {err1=} {err2=}")

    err1 = np.sqrt(err1/len(mat))
    err2 = np.sqrt(err2/len(mat))

    str1 += f"SCORE {p(err1)} {err1:.3f} "
    str2 += f"SCORE {p(err2)} {err2:.3f} "

    out1 = ''.join([t[0] for t in digits1])
    out2 = ''.join([t[0] for t in digits2])

    frame = meta.get('frame')
    stat = "stat_" + name

    if out1 == "314159":
        meta.inc(f"{stat}tmatch")
    elif '#' in out1:
        meta.inc(f"{stat}trej")
    else:
        #log(f"Err thresh {name} {frame=}: {str1}")
        meta.inc(f"{stat}terr")


    if p(err2) > cutoff:
        meta.set("result", out2)
        if out2 == "314159":
            meta.inc(f"{stat}match")
            meta.append(stat+"mscores", p(err2))
            #print(f"{frame};2;{p(err2)};{out2}")
        else:
            #meta.set("save")
            #log(f"Err {name} {frame=}: {str2}")
            meta.inc(f"{stat}err")
            meta.append(stat+"escores", p(err2))
            #print(f"{frame};1;{p(err2)};{out2}")
    else:
        meta.inc(f"{stat}rej")
        #log(f"Rej {name} {frame=}: {str2}")
        meta.set("result", "REJ"+out2)
        #print(f"{frame};0;{p(err2)};{out2}")


    dbg(f"Number1: {str1}")
    dbg(f"Number2: {str2}")

    dbg(f"Number: {out2} {out1}")
    #log(f"Number rmsd: {out2}")
    return digits2


# normalize to range [0,1]
def normalize_matrix(matrix, binary=False):
    first = matrix.flat[0]
    # short circuit on one value matrix
    if np.all(matrix==first):
        return None

    max = 0
    if binary:
        max = meta.get('maxrect')
    else:
        min = np.min(matrix)
        matrix = matrix - min
        max = np.max(matrix)

    matrix = matrix / max

    if meta.get("invertDigits"): # invert
        matrix = 1 - matrix

    return matrix

def threshold_matrix(l, thresh):
    if meta.get("invertDigits"): # invert
        thresh = 1 - thresh
    # Use Otsu's value as threshold
    l = (l > thresh).astype("int")
    return l

def digits(font=0):
    a = font
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
    return digits



def mse_match(digit, threshold=0.5, font=0, binary=False):
    assert threshold == 0.5 # other values later
    if isdbg: dbg(f"IN1: {digit}")

    digit = normalize_matrix(digit, binary)
    if digit is None:
        return '#', 1

    result = '#'
    #if isdbg: dbg(f"IN: {digit}")

    minmsd = 100
    minrmsd = 100

    for j, ref in enumerate(digits(font)):
        #dbg(f"DIG: {ref}")

        dev = (digit-ref)**2
        sum = np.sum(dev)
        msd = sum/7
        rmsd = np.sqrt(msd)

        #score = f"{100*(1-rmsd):3.0f}"
        #dbg(f"CAND: {j} {p(dev2)} {dev2:.3f} {dev5:.3f}, {p(dev3)} {p(dev4)}")
        #if isdbg: dbg(f"CAND: {j} {p(dev2/7)} {dev2:.2f} {rmsd:.2f} {p(rmsd)}")
        if rmsd < minrmsd:
            minrmsd = rmsd
            minmsd = msd
            result = str(j)
        #dbg(f"ERR {j} {dev2:.3f} {dev}")

    #confidence = 1 - (minerr / 7)

    if isdbg: dbg(f"RMSD:   {result} {p(minrmsd)} {minrmsd:.3f}")
    return (result, minrmsd)

def thresh_match(digit, threshold=0.5, font=0, binary=False):
    digit = normalize_matrix(digit, binary)
    if digit is None:
        return '#', 1

    orig = digit
    digit = threshold_matrix(digit, threshold)

    result = '#'
    diff = np.ones_like(digit)
    for j, ref in enumerate(digits(font)):
        if np.array_equal(ref, digit):
            result=str(j)
            diff = abs(orig - ref)
            break

    confidence = np.sum(diff)/7
    if isdbg: dbg(f"THRESH: {result} {p(confidence)} {confidence:.3f}")
    return (result, confidence)
