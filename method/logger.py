import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def setup_logger(log_folder, console_level):
    # Create the file handler for logging to a file
    log_filename = os.path.join(
        log_folder,
        f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Create the stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(console_level)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.DEBUG)


def assert_and_log(condition, message):
    if not condition:
        logger.error(f"Assertion failed: {message}")
        raise AssertionError(message)


def raise_and_log(exception, message):
    logger.error(f"{exception.__name__}: {message}")
    raise exception(message)
