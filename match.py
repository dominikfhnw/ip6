import log
import meta
import numpy as np

log, dbg, logger = log.auto(__name__)

def process(mat, threshold):
    dbg("started match")

    digits = []
    digits2 = []

    for i, digit in enumerate(mat):
        dbg(f"NUM: {i}")
        result, confidence = thresh_match(digit)
        result2, confidence2, rmsd = mse_match(digit)

        digits.append(result)
        digits2.append((result2,confidence2))

    out = ''.join(digits)
    out2 = ''.join([t[0] for t in digits2])

    meta.set("result", out)
    log(f"Number: {out}")
    log(f"Number2: {out2}")
    return digits2

# normalize to range [0,1]
def normalize_matrix(matrix):
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


def p(float):
    #return f"{100*(1-float):3.0f}"
    if float != float:
        log("GOT UNEXPECTED NaN")
        return 999
    return int(100*(1-float))

def mse_match(digit, threshold=0.5, font=0):
    assert threshold == 0.5 # other values later
    digit = normalize_matrix(digit)

    result = '#'
    #dbg(f"IN: {digit}")

    minerr = 100

    for j, ref in enumerate(digits(font)):
        #dbg(f"DIG: {ref}")

        dev = (digit-ref)**2
        dev2 = np.sum(dev)
        #dev3 = np.sqrt(dev2)
        #dev4 = np.sum(np.sqrt(dev))
        rmsd = np.sqrt(dev2/7)
        #score = f"{100*(1-rmsd):3.0f}"
        #dbg(f"CAND: {j} {p(dev2)} {dev2:.3f} {dev5:.3f}, {p(dev3)} {p(dev4)}")
        dbg(f"CAND: {j} {p(dev2/7)} {dev2:.2f} {rmsd:.2f} {p(rmsd)}")
        if rmsd < minerr:
            minerr = rmsd
            result = str(j)
        #dbg(f"ERR {j} {dev2:.3f} {dev}")

    #confidence = 1 - (minerr / 7)
    dbg(f"MSE:  {result} {p(minerr)} {minerr:.3f}")
    return (result, p(minerr), minerr)

def thresh_match(digit, threshold=0.5, font=0):
    digit = normalize_matrix(digit)
    digit = threshold_matrix(digit, threshold)

    result = '#'
    confidence = 0
    for j, ref in enumerate(digits(font)):
        if np.array_equal(ref, digit):
             result=str(j)
    dbg(f"THRESH: {result} ({100*confidence}%) {digit}")
    return (result, confidence)
