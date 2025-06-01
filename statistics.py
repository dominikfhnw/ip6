import numpy as np

import log
import meta
import atexit

log, dbg, logger, isdbg = log.auto2(__name__)

def process():
    log("nothing much to do")

def stats(name, mat):
    if mat is None: return
    min = np.min(mat)
    max = np.max(mat)
    mean = np.mean(mat)
    std = np.std(mat)

    logger.warn(f"{name}: {min=} {max=} {mean=:.2f} {std=:.2f}")


def show(name):
    stat = 'stat_'+name

    file = meta.get("filename").removeprefix("benchmark/")
    avgoption = meta.get("ocrComposite")
    avg = ""
    if avgoption:
        avg = "avg"
    else:
        avg = "noavg"

    match = meta.getnum(stat+'match')
    rej = meta.getnum(stat+'rej')
    err = meta.getnum(stat+'err')

    tmatch = meta.getnum(stat+'tmatch')
    trej = meta.getnum(stat+'trej')
    terr = meta.getnum(stat+'terr')

    mscores = meta.getnum(stat+'mscores')
    escores = meta.getnum(stat+'escores')
    #logger.warn(f"{name}: {match=} {err=} {rej=}    {tmatch=} {terr=} {trej=}")

    mmin = np.min(mscores)
    mmax = np.max(mscores)
    mmean = np.mean(mscores)
    mstd = np.std(mscores)

    emin = np.min(escores)
    emax = np.max(escores)
    emean = np.mean(escores)
    estd = np.std(escores)

    #logger.warn(f"{file};{name};RMSD;{avg};{match};{err};{rej};{mmin};{mmax};{mmean};{mstd};{emin};{emax};{emean};{estd}")
    #logger.warn(f"{file};{name};THRESH;{avg};{tmatch};{terr};{trej}")


    #stats(name+' match', mscores)
    #stats(name+' err', escores)
    mtxt = ";".join(map(str, mscores))
    if escores:
        etxt = ";".join(map(str, escores))
    else:
        etxt = ""
    #logger.warn(f"{file};{name};error;{etxt}")
    #logger.warn(f"{file};{name};match;{mtxt}")
    if True:
        if mscores:
            for i in mscores:
                logger.warn(f"{file};{name};match;{i}")
        else:
            logger.warn(f"{file};{name};match;")
        if escores:
            for i in escores:
                logger.warn(f"{file};{name};error;{i}")
        else: logger.warn(f"{file};{name};error;")


#logger.warn(f"mscores {mscores}")
    #logger.warn(f"escores {escores}")


def _end():
    ft = len(meta.get("ft"))
    fsum = sum(meta.get("ft")[-ft:])
    fps = ft / fsum
    time = fsum / ft
    logger.warn(f"Overall FPS: {fps:.2f}, average frame time {1000*time:.2f}ms")
    ocr = meta.get('ocr_methods')
    if not ocr:
        dbg("no OCR done")
        return
    source = meta.get("source")
    logger.warn(f"SOURCE: {source}")
    frames = meta.get('frame')
    one = meta.get('stat_onemarker')
    both = meta.get('stat_bothmarker')

    logger.warn('')
    logger.warn(f"{frames=} {one=} {both=}")
    for name in sorted(set(ocr)):
        #logger.warn('')
        show(name)
    logger.warn('')

atexit.register(_end)