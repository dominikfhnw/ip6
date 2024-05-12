import logging
import meta

logger = logging
#logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

logging.basicConfig(
    #format='%(asctime)s: %(message)s',
    #format='%(asctime)s.%(msecs)03d %(levelname)s:%(module)s: %(message)s',
    format='%(asctime)s:%(module)s: %(message)s',
    #datefmt='%Y-%m-%d %H:%M:%S',
    level=meta.get("logLevel")
)
logger.info("logger started")

def init(name):
    new = logger.getLogger(name)
    if name == "__main__":
        logger.info("main module initialized")
    else:
        logger.info(f'loaded module "{name}"')
    return new

def auto(name):
    logger = init(name)
    return (logger.info, logger.debug, logger)