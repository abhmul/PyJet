from . import data
from . import utils

import logging
import time
import queue
import threading
import copy


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
                steps_per_epoch=generator.steps_per_epoch,
                batch_size=generator.batch_size,
            )
        else:
            logging.warning(
                "Input generator does not have a steps_per_epoch or batch_size "
                "attribute. Continuing without them."
            )
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
        self.epoch_logs[metric.__name__] = metric.accumulate()

    def on_epoch_end(self):
        for metric_name, score in self.epoch_logs.items():
            self.setdefault(metric_name, []).append(score)

    def log_validation_metric(self, metric):
        self.epoch_logs["val_" + metric.__name__] = metric.accumulate()


class LossManager(object):
    @utils.resettable
    def __init__(self):
        self.__loss_names = []
        self.__loss_input_dict = {}
        self.__loss_weight_dict = {}
        self.__loss_dict = {}
        self.__verify_loss_args = True
        self.__loss_scores = {}

    def __len__(self):
        return len(self.__loss_names)

    @property
    def names(self):
        return list(self.__loss_names)

    def _compute_single_loss(self, model, targets, name):
        # Cache the score for logging
        self.__loss_scores[name] = self.__loss_weight_dict[name] * self.__loss_dict[
            name
        ](
            *[
                getattr(model, loss_input)
                for loss_input in self.__loss_input_dict[name]
            ],
            targets
        )
        return self.__loss_scores[name]

    def verify_args(self, model):
        for loss_name, loss_inputs in self.__loss_input_dict.items():
            for loss_input in loss_inputs:
                if not hasattr(model, loss_input):
                    raise AttributeError(
                        "Model does not have attribute {loss_input}, which"
                        " is an input for the loss {loss_name}".format(
                            loss_input=loss_input, loss_name=loss_name
                        )
                    )

    def loss(self, model, targets):
        # This means we need to verify that the input arguments for the loss
        # exist, and notify the user if they don't
        if self.__verify_loss_args:
            self.verify_args(model)
            self.__verify_loss_args = False

        # Compute the loss
        return sum(
            self._compute_single_loss(model, targets, loss_name)
            for loss_name in self.__loss_names
        )

    def get_loss_score(self, name=None):
        if name is None:
            assert not len(self.__loss_names), (
                "Need to specify a loss if " "using multiple losses."
            )
            name = self.__loss_names[0]
        return self.__loss_scores[name]

    def add_loss(self, loss_fn, inputs, weight=1.0, name=None):
        if name is None:
            name = "loss_{}".format(len(self.__loss_dict))
        self.__loss_dict[name] = loss_fn
        self.__loss_input_dict[name] = inputs
        self.__loss_weight_dict[name] = weight
        self.__loss_names.append(name)

    def remove_loss(self, name=None):
        if name is None:
            name = self.__loss_names.pop()
        else:
            self.__loss_names.remove(name)
        loss_fn = self.__loss_dict.pop(name)
        inputs = self.__loss_input_dict.pop(name)
        weight = self.__loss_weight_dict.pop(name)
        return {"name": name, "loss": loss_fn, "inputs": inputs, "weight": weight}

    def clear_losses(self):
        self.reset()


class OptimizerManager(object):
    @utils.resettable
    def __init__(self):
        self.__optimizer_names = []
        self.__optimizer_dict = {}

    def __len__(self):
        return len(self.__optimizer_names)

    @property
    def names(self):
        return list(self.__optimizer_names)

    @property
    def optimizers(self):
        return list(self.__optimizer_dict.values())

    def add_optimizer(self, optimizer, name=None):
        if name is None:
            name = "optimizer_{}".format(len(self))
        self.__optimizer_dict[name] = optimizer
        self.__optimizer_names.append(name)

    def remove_optimizer(self, name=None):
        if name is None:
            name = self.__optimizer_names.pop()
        else:
            self.__optimizer_names.remove(name)
        optimizer = self.__optimizer_dict.pop(name)
        return {"name": name, "optimizer": optimizer}

    def clear_optimizers(self):
        self.reset()
