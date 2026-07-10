import logging

import ctranslate2


def configure_logging() -> None:
    ctranslate2.set_log_level(logging.WARN)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s -%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # The root DEBUG level above is meant for our own modules; without this these
    # libraries inherit it and log a record per HTTP connection open and close.
    for name in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)
