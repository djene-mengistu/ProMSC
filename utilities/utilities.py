
import os
import logging
import sys
import warnings
import random
import numpy as np
warnings.filterwarnings("ignore")


#Creating the logging functions
def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_logger(log_dir):
    create_exp_dir(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'run.log'), encoding='utf-8')
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Nas Seg')
    logger.addHandler(fh)
    return logger
