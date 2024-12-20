import logging

import ctranslate2


def configure_logging():

    ctranslate2.set_log_level(logging.WARN)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s -%(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
