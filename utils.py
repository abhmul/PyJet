from collections import defaultdict
import tqdm

def log(stmt, verbosity, log_verbosity=1):
    if verbosity >= log_verbosity:
        print(stmt)

class ProgBar(object):

    def __init__(self, verbosity=1):
        self.tqdm = tqdm.tqdm if verbosity == 1 else lambda x, postfix: x
        self.stat_sums = defaultdict(float)
        self.stat_counts = defaultdict(int)
        self.postfix = defaultdict(float)

    def update(self, name, val):
        if verbosity == 0:
            return
        self.stat_sums[name] += val
        self.stat_counts[name] += 1
        self.postfix[name] = self.stat_sums[name] / self.stat_counts[name]

    def __call__(self, iterable):
        return self.tqdm(iterable, postfix=self.postfix)
