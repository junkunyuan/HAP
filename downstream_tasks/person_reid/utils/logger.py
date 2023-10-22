import logging
import os
import sys


def setup_logger(output_dir, local_rank, name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = '[%(asctime)s %(filename)s:%(lineno)s] %(message)s'
    datefmt = f'%Y-%m-%d %H:%M:%S'

    if local_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_rank{local_rank}.txt'), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt))
    logger.addHandler(file_handler)

    return logger
