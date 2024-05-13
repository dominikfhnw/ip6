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

    match = meta.get(stat+'match')
    rej = meta.get(stat+'rej')
    err = meta.get(stat+'err')

    tmatch = meta.get(stat+'tmatch')
    trej = meta.get(stat+'trej')
    terr = meta.get(stat+'terr')

    mscores = meta.get(stat+'mscores')
    escores = meta.get(stat+'escores')

    logger.warn(f"{name}: {match=} {err=} {rej=}    {tmatch=} {terr=} {trej=}")
    stats(name+' match', mscores)
    stats(name+' err', escores)

    #logger.warn(f"mscores {mscores}")
    #logger.warn(f"escores {escores}")


def _end():
    frames = meta.get('frame')
    one = meta.get('stat_onemarker')
    both = meta.get('stat_bothmarker')

    logger.warn('')
    logger.warn(f"{frames=} {one=} {both=}")
    for name in sorted(set(meta.get('ocr_methods'))):
        logger.warn('')
        show(name)
    logger.warn('')

atexit.register(_end)