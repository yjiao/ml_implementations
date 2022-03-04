"""

Various utility functions used for training models.

"""

import sys
with open("include_path.txt") as f:
    path = f.readline()
sys.path.append(path)

import numpy as np
import tensorflow as tf


def get_num_gpus() -> int:
    return len(tf.config.experimental.list_physical_devices("GPU"))


def lr_schedule(
    cur_it: int, max_it: int, max_warm_it: int = 300, max_lr: int = 0.001
) -> float:
    """Cosine learning rate schedule.

    cur_it: current iteration
    max_it: maximum iteration
    max_warm_it: iteration at which we should stop warming
    max_lr: max learning rate

    """
    if cur_it < max_warm_it:
        lr = cur_it / max_warm_it * max_lr
    else:
        x = cur_it / max_it * np.pi
        lr = max_lr * (np.cos(x) + 1) / 2
    return lr
