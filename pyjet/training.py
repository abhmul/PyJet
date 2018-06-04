from . import data

import logging
import time
import queue
import threading


class GeneratorEnqueuer(data.BatchGenerator):
    """Builds a queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which endlessly yields data
    """

    def __init__(self, generator):
        # Copy the steps per epoch and batch size if it has one
        if hasattr(generator, "steps_per_epoch") and hasattr(
                generator, "batch_size"):
            super(GeneratorEnqueuer, self).__init__(
                steps_per_epoch=generator.steps_per_epoch,
                batch_size=generator.batch_size)
        else:
            logging.warning(
                "Input generator does not have a steps_per_epoch or batch_size "
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
                self._threads.append(
                    threading.Thread(target=data_generator_task))
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
            raise ValueError(
                "Generator must be running before iterating over it")
        while True:
            if not self.queue.empty():
                return self.queue.get()
            else:
                # print("Waiting...")
                time.sleep(self.wait_time)


class TrainingLogs(dict):
    def __init__(self):
        super().__init__()
        self.epoch_logs = {}
        self.batch_logs = {}

    def on_epoch_begin(self):
        self.epoch_logs = {}
        self.batch_logs = {}

    def log_metric(self, metric, score):
        self.batch_logs[metric.__name__] = score.item()
        self.epoch_logs[metric.__name__] = metric.accumulate().item()

    def on_epoch_end(self):
        for metric_name, score in self.epoch_logs.items():
            self.setdefault(metric_name, []).append(score)

    def log_validation_metric(self, metric):
        self.epoch_logs["val_" + metric.__name__] = metric.accumulate().item()
