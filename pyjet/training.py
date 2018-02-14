from . import backend as J
from . import data

import logging
import time
import queue
import threading
from collections import defaultdict
import numpy as np
from tqdm import trange
from torch.autograd import Variable


class GeneratorEnqueuer(data.BatchGenerator):
    """Builds a queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which endlessly yields data
    """

    def __init__(self, generator):
        # Copy the steps per epoch and batch size if it has one
        if hasattr(generator, "steps_per_epoch") and hasattr(generator, "batch_size"):
            super(GeneratorEnqueuer, self).__init__(
                steps_per_epoch=generator.steps_per_epoch, batch_size=generator.batch_size)
        else:
            logging.warning("Input generator does not have a steps_per_epoch or batch_size "
                            "attribute. Continuing without them.")
        self._generator = generator
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.wait_time = None

    def start(self, workers=1, max_q_size=10, wait_time=0.05):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """
        self.wait_time = wait_time

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise
        try:
            self.queue = queue.Queue()
            self._stop_event = threading.Event()

            for _ in range(workers):
                self._threads.append(threading.Thread(target=data_generator_task))
                self._threads[-1].start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().
        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout)

        self._threads = []
        self._stop_event = None
        self.queue = None

    def __next__(self):
        if not self.is_running():
            raise ValueError("Generator must be running before iterating over it")
        while True:
            if not self.queue.empty():
                return self.queue.get()
            else:
                # print("Waiting...")
                time.sleep(self.wait_time)

# A simple object for logging


class MetricLogs(object):
    def __init__(self, metrics):
        self.logs = {}
        self._init_keys([metric.__name__ for metric in metrics])
        self.metrics = metrics

    def __len__(self):
        return len(self.logs)

    def items(self):
        return self.logs.items()

    def values(self):
        return self.logs.values()

    def __iter__(self):
        return iter(self.logs)

    def _init_keys(self, metric_names):
        for metric_name in metric_names:
            self.logs[metric_name] = []

    def _update_key(self, key, value):
        self.logs[key].append(value)

    def _update_keys(self, metric_names, scores):
        for score, metric_name in zip(scores, metric_names):
            self._update_key(metric_name, score)

    def update_log(self, stat, value):
        self._update_key(stat.__name__, value)

    def update_logs(self, metrics, scores):
        self._update_keys([metric.__name__ for metric in metrics], scores)

    def average_key(self, key):
        return sum(self.logs[key]) / len(self.logs[key])

    def average(self, stat):
        return self.average_key(stat.__name__)

    def __getitem__(self, key):
        return self.logs[key]

    def get(self, key):
        return self.logs.get(key)

    # TODO figure out where this is ueds since its technically wrong
    def keys(self):
        return self.metrics

# A simple class for a progress bar


class ProgBar(object):

    def __init__(self, verbosity=1):
        self.tqdm = trange
        self.stat_sums = defaultdict(float)
        self.stat_counts = defaultdict(int)
        self.postfix = defaultdict(float)
        self.verbosity = verbosity

    def update_stat(self, name, val):
        self.stat_sums[name] += val
        self.stat_counts[name] += 1
        self.postfix[name] = self.stat_sums[name] / self.stat_counts[name]

    def update_stat_from_func(self, func, val, prefix=""):
        self.update_stat(prefix + func.__name__, val)

    def update_stats(self, names, values):
        for name, value in zip(names, values):
            self.update_stat(name, value)

    def update_stats_from_func(self, funcs, values, prefix=""):
        self.update_stats([prefix + func.__name__ for func in funcs], values)

    def update_bar(self):
        if self.verbosity > 0:
            self.tqdm.set_postfix(self.postfix)

    def __call__(self, high):
        if self.verbosity > 0:
            self.tqdm = self.tqdm(high)
            return self.tqdm
        return range(high)
