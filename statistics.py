import numpy as np

import log
import meta
import atexit

log, dbg, logger, isdbg = log.auto2(__name__)

def process():
    log("nothing much to do")

def stats(name, mat):
    min = np.min(mat)
    max = np.max(mat)
    mean = np.mean(mat)
    std = np.std(mat)

    logger.warn(f"{name}: {min=} {max=} {mean=:.2f} {std=:.2f}")


def show(name):
    stat = 'stat_'+name
    match = meta.get(stat+'match')
    err = meta.get(stat+'err')

    mscores = meta.get(stat+'mscores')
    escores = meta.get(stat+'escores')

    logger.warn(f"{name}: {match=} {err=}")
    stats(name+' match', mscores)
    stats(name+' err', escores)

    #logger.warn(f"mscores {mscores}")
    logger.warn(f"escores {escores}")


def _end():
    frames = meta.get('frame')
    one = meta.get('stat_onemarker')
    both = meta.get('stat_bothmarker')

    logger.warn('')
    logger.warn(f"{frames=} {one=} {both=}")
    logger.warn('')
    show("std")
    logger.warn('')
    show("gauss")
    logger.warn('')

atexit.register(_end)