import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')


def setup_logger(log_folder, level):
    # Create the file handler for logging to a file
    log_filename = os.path.join(
        log_folder, 
        f'log_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    # Create the stream handler for logging to the console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(level.upper())
