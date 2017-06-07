from collections import defaultdict
from tqdm import trange
import numpy as np

def log(stmt, verbosity, log_verbosity=1):
    if verbosity >= log_verbosity:
        print(stmt)

class ProgBar(object):

    def __init__(self, verbosity=1):
        self.tqdm = trange if verbosity == 1 else range
        self.stat_sums = defaultdict(float)
        self.stat_counts = defaultdict(int)
        self.postfix = defaultdict(float)
        self.verbosity = verbosity

    def update(self, name, val):
        if self.verbosity == 0:
            return
        self.stat_sums[name] += val
        self.stat_counts[name] += 1
        self.postfix[name] = self.stat_sums[name] / self.stat_counts[name]
        self.tqdm.set_postfix(dict(self.postfix))

    def __call__(self, high):
        self.tqdm = self.tqdm(high)
        return self.tqdm

def to_categorical(y, num_classes=None):
    """ Copied from Keras :'(
    Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical
