# Python standard library modules
import copy
import os
import logging
import threading
import time
import queue
from collections import deque
import warnings
import itertools
from functools import partial

# Third party libraries
import numpy as np
from tqdm import tqdm
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable  # DO NOT REMOVE
print("Using PyTorch version:", torch.__version__)


### UTILS ###
def resettable(f):
    """
    Decorator to make a python object resettable. Note that this will
    no work with inheritance. To reset an object, simply call its reset
    method.
    """
    def __init_and_copy__(self, *args, **kwargs):
        f(self, *args)

        def reset(o=self):
            o.__dict__ = o.__original_dict__
            o.__original_dict__ = copy.deepcopy(self.__dict__)

        self.reset = reset
        self.__original_dict__ = copy.deepcopy(self.__dict__)

    return __init_and_copy__


def safe_open_dir(dirpath):
    if not os.path.isdir(dirpath):
        logging.info("Directory %s does not exist, creating it" % dirpath)
        os.makedirs(dirpath)
    return dirpath


### BACKEND ###
# Define the global backend variables
epsilon = 1e-11
channels_mode = "channels_first"
logging.info(f"PyJet using config: epsilon={epsilon}, channels_mode"
             "={channels_mode}")

# Set up the use of cuda if available
use_cuda = torch.cuda.is_available()


def cudaFloatTensor(x):
    # Optimization for casting things to cuda tensors
    return torch.FloatTensor(x).cuda()


def cudaLongTensor(x):
    return torch.LongTensor(x).cuda()


def cudaByteTensor(x):
    return torch.ByteTensor(x).cuda()


def cudaZeros(*args):
    return torch.zeros(*args).cuda()


def cudaOnes(*args):
    return torch.ones(*args).cuda()


def flatten(x):
    """Flattens along axis 0 (# rows in == # rows out)"""
    return x.view(x.size(0), -1)


def softmax(x):
    # BUG some shape error
    # .clamp(epsilon, 1.)
    normalized_exp = (x - x.max(1)[0].expand(*x.size())).exp()
    return normalized_exp / normalized_exp.sum(1).expand(*x.size())


def batch_sum(x):
    """Sums a tensor long all non-batch dimensions"""
    return x.sum(tuple(range(1, x.dim())))


def zero_center(x):
    return x - x.mean()


def standardize(x):
    std = (x.pow(2).mean() - x.mean().pow(2)).sqrt()
    return zero_center(x) / std.expand(*x.size()).clamp(min=epsilon)


def from_numpy(x):
    return torch.from_numpy(x).cuda() if use_cuda else torch.from_numpy(x)


def to_numpy(x):
    return x.cpu().numpy() if use_cuda else x.numpy()


def arange(start, end=None, step=1, out=None):
    if end is None:
        x = torch.arange(0, start, step=step, out=out)
    else:
        x = torch.arange(start, end, step=step, out=out)
    return x.cuda() if use_cuda else x


def rand(*sizes, out=None):
    x = torch.rand(*sizes, out=out)
    return x.cuda() if use_cuda else x


# use_cuda = False
FloatTensor = cudaFloatTensor if use_cuda else torch.FloatTensor
LongTensor = cudaLongTensor if use_cuda else torch.LongTensor
ByteTensor = cudaByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
# Tensor fillers
zeros = cudaZeros if use_cuda else torch.zeros
ones = cudaOnes if use_cuda else torch.ones

print("PyJet is using " + ("CUDA" if use_cuda else "CPU") + ".")

### DATA ###
# TODO Create a dataset for HDF5 and Torch Tensor

# VERBOSITY = namedtuple(
#     'VERBOSITY', ['QUIET', 'NORMAL', 'VERBOSE', 'DEBUG'])(0, 1, 2, 3)


class Dataset(object):
    """
    An abstract container for data designed to be passed to a model.
    This container should implement create_batch. It is only necessary
    to implement validation_split() if you use this module to split your
    data into a train and test set. Same goes for kfold()

    # Note:
        Though not forced, a Dataset is really a constant object. Once created,
        it should not be mutated in any way.
    """

    def __init__(self, *args, **kwargs):
        # self.verbosity = verbosity
        pass

    def __len__(self):
        """The length is used downstream by the generator if it is not inf."""
        return float('inf')

    # def log(self, statement, verbosity):
    #     if self.verbosity >= verbosity:
    #         print(statement)

    def create_batch(self, *args, **kwargs):
        """
        This method creates a batch of data to be sent to a model.

        Returns:
            A batch in the form of any type that can be cast to torch tensor
            by a model (numpy, HDF5, torch tensor, etc.).
        """
        raise NotImplementedError()

    def flow(self,
             steps_per_epoch=None,
             batch_size=None,
             shuffle=True,
             replace=False,
             seed=None):
        """
        This method creates a generator for the data.

        Returns:
            A DatasetGenerator with settings determined by inputs to this
            method that generates batches made by this dataset
        """
        return DatasetGenerator(
            self,
            steps_per_epoch=steps_per_epoch,
            batch_size=batch_size,
            shuffle=shuffle,
            replace=replace,
            seed=seed)

    def validation_split(self, split=0.2, **kwargs):
        raise NotImplementedError()

    def kfold(self, k, **kwargs):
        raise NotImplementedError()


class BatchGenerator(object):
    """
    An abstarct iterator to create batches for a model.

    # Arguments:
        steps_per_epoch -- The number of iterations in one epoch (optional)
        batch_size -- The number of samples in one batch
    """

    def __init__(self, steps_per_epoch=None, batch_size=None):
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError()


class DatasetGenerator(BatchGenerator):
    """
    An iterator to create batches for a model using a Dataset. 2 of the
    following must be defined
        -- The input Dataset's length
        -- steps_per_epoch
        -- batch_size
    Also, if the Dataset's length is not defined, its create_batch method
    should not take any inputs

    # Arguments
        dataset -- the dataset to generate from
        steps_per_epoch -- The number of iterations in one epoch (optional)
        batch_size -- The number of samples in one batch (optional)
        shuffle -- Whether or not to shuffle the dataset before each epoch
                   default: True
        replace -- Whether or not to sample with replacement. default: False
        seed -- A seed for the random number generator (optional).
    """

    def __init__(self,
                 dataset,
                 steps_per_epoch=None,
                 batch_size=None,
                 shuffle=True,
                 replace=False,
                 seed=None):
        super(DatasetGenerator, self).__init__(steps_per_epoch, batch_size)
        self.dataset = dataset
        self.shuffle = shuffle
        self.replace = replace
        self.seed = seed
        self.index_array = None
        self.lock = threading.Lock()

        # Some input checking
        check = (int(self.steps_per_epoch is not None) +
                 int(self.batch_size is not None) +
                 int(len(self.dataset) != float("inf")))
        if check < 2:
            raise ValueError(
                "2 of the following must be defined: len(dataset),"
                " steps_per_epoch, and batch_size.")
        # Otherwise, we're good, infer the missing info
        if len(self.dataset) != float('inf'):
            self.index_array = np.arange(len(self.dataset))
        if self.batch_size is None:
            if self.steps_per_epoch is None:
                raise ValueError()
            self.batch_size = int(
                (len(self.dataset) + self.steps_per_epoch - 1) /
                self.steps_per_epoch)
        if self.steps_per_epoch is None:
            self.steps_per_epoch = int(
                (len(self.dataset) + self.batch_size - 1) / self.batch_size)
        # Set the seed if we have one
        if self.seed is not None:
            np.random.seed(self.seed)
        self.batch_argument_generator = self.create_batch_argument_generator()

    def create_batch_argument_generator(self):
        """
        This is an iterator that generates the necessary arguments needed to
        create each batch. By default, it will generate indicies from an index
        array

        # Note:
            This will raise a NotImplementedError if we don't have an index
            array, since we can't generate batch indicies if we don't know
            what our indicies are. If your dataset does not have indicies,
            you'll have to implement this yourself.

            If you implement this yourself, note that the output must be an
            iterable of all the arguments your dataset's create_batch method
            needs.
        """
        if self.index_array is None:
            raise NotImplementedError()
        while True:
            # Shuffle if we need to
            if self.shuffle:
                np.random.shuffle(self.index_array)
            for i in range(0, len(self.index_array), self.batch_size):
                if self.replace:
                    yield (np.random.choice(self.index_array, self.batch_size,
                                            True), )
                else:
                    yield (self.index_array[i:i + self.batch_size], )

    def __next__(self):
        # This is a critical section, so we lock when we need the next indicies
        if self.index_array is not None:
            with self.lock:
                batch_arguments = next(self.batch_argument_generator)
        else:
            batch_arguments = tuple([])
        # print("Batch Arguments: ", batch_arguments[0])
        return self.dataset.create_batch(*batch_arguments)

    def toggle_shuffle(self):
        self.shuffle = not self.shuffle

    def restart(self):
        self.batch_argument_generator = self.create_batch_argument_generator()


class BatchPyGenerator(BatchGenerator):
    """
    A BatchGenerator that generates using a python iterator.

    # Arguments:
        pygen -- the python iterator from which to generate batches
        steps_per_epoch -- The number of iterations in one epoch
    """

    def __init__(self, pygen, steps_per_epoch):
        super(BatchPyGenerator, self).__init__(steps_per_epoch)
        self.pygen = pygen

    def __iter__(self):
        return self.pygen

    def __next__(self):
        return next(self.pygen)


class NpDataset(Dataset):
    """
    A Dataset that is built from numpy data.

    # Arguments
        x -- The input data as a numpy array
        y -- The target data as a numpy array (optional)
    """

    # TODO define the kfold method for NpDataset

    def __init__(self, x, y=None, ids=None):
        super(NpDataset, self).__init__()

        self.x = x
        self.y = y
        self.ids = ids

        assert isinstance(self.x, np.ndarray), "x must be a numpy array."
        if self.y is not None:
            assert isinstance(self.y, np.ndarray), "y must be a numpy array " \
                "or None."
        if self.ids is not None:
            assert isinstance(self.ids, np.ndarray), "ids must be a numpy " \
                "or None."

        self.output_labels = self.has_labels
        if self.has_labels:
            assert len(self.x) == len(
                self.y), ("Data and labels must have same number of" +
                          "samples. X has shape ", len(x), " and Y has shape ",
                          len(y), ".")
        if self.has_ids:
            assert len(self.x) == len(
                self.ids), ("Data and ids must have same number of" +
                            "samples. X has shape ", len(x),
                            " and ids has shape ", len(ids), ".")

    def __len__(self):
        return len(self.x)

    @property
    def has_ids(self):
        return self.ids is not None

    @property
    def has_labels(self):
        return self.y is not None

    def toggle_labels(self):
        self.output_labels = not self.output_labels

    def create_batch(self, batch_indicies):
        outputs = [
            self.x[batch_indicies],
        ]
        if self.output_labels:
            outputs.append(self.y[batch_indicies])
        if not self.output_labels:
            return outputs[0]
        return outputs[0], outputs[1]

    @staticmethod
    def get_stratified_split_indicies(split, shuffle, seed, stratify_by):
        if shuffle:
            if seed is not None:
                np.random.seed(seed)

        # Get all the unique output labels
        unq_labels = np.unique(stratify_by, axis=0)
        val_splits = []
        train_splits = []
        for unq_label in unq_labels:
            # Find where the entire output label matches the unique label
            if stratify_by.ndim == 1:
                label_mask = stratify_by == unq_label
            else:
                non_batch_dims = tuple(range(1, stratify_by.ndim))
                label_mask = np.all(stratify_by == unq_label,
                                    axis=non_batch_dims)
            # Get the indicies where the label matches
            label_inds = np.where(label_mask)[0]
            if shuffle:
                np.random.shuffle(label_inds)
            split_ind = int(split * len(label_inds))
            val_splits.append(label_inds[:split_ind])
            train_splits.append(label_inds[split_ind:])
        train_split = np.concatenate(train_splits, axis=0)
        val_split = np.concatenate(val_splits, axis=0)
        # Shuffle one more time to get the labels shuffled
        if shuffle:
            np.random.shuffle(train_split)
            np.random.shuffle(val_split)
        return train_split, val_split

    def get_split_indicies(self, split, shuffle, seed, stratified,
                           stratify_by):
        if stratified:
            if stratify_by is None:
                stratify_by = self.y
            assert stratify_by is not None, "Data must have labels to " \
                                            "stratify by."
            assert len(stratify_by) == len(self), "Labels to stratify by " \
                "have same length as the dataset."

        if shuffle:
            if seed is not None:
                np.random.seed(seed)

        if stratified:
            train_split, val_split = self.get_stratified_split_indicies(
                split, shuffle, seed, stratify_by)
        else:
            # Default technique of splitting the data
            split_ind = int(split * len(self))
            val_split = slice(split_ind)
            train_split = slice(split_ind, None)
            if shuffle:
                indicies = np.random.permutation(len(self))
                train_split = indicies[train_split]
                val_split = indicies[val_split]

        return train_split, val_split

    def get_kfold_indices(self, k, shuffle, seed):
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            indicies = np.random.permutation(len(self))
        else:
            indicies = np.arange(len(self))

        for i in range(k):
            split = 1.0 / k
            # Default technique of splitting the data
            split_start = int(i * split * len(self))
            split_end = int((i + 1) * split * len(self))
            val_split = slice(split_start, split_end)
            train_split_a = indicies[0:split_start]
            train_split_b = indicies[split_end:]
            if shuffle:
                train_split_a = indicies[train_split_a]
                train_split_b = indicies[train_split_b]
                val_split = indicies[val_split]

            yield np.concatenate([train_split_a, train_split_b]), val_split

    def validation_split(self,
                         split=0.2,
                         shuffle=False,
                         seed=None,
                         stratified=False,
                         stratify_by=None):
        """
        Splits the NpDataset into two smaller datasets based on the split

        Args:
            split (float, optional): The fraction of the dataset to make
                validation. Defaults to 0.2.
            shuffle (bool, optional): Whether or not to randomly sample the
                validation set and train set from the parent dataset. Defaults
                to False.
            seed (int, optional): A seed for the random number generator.
                Defaults to None.
            stratified (bool, optional): Whether or not to sample the
                validation set to have the same label distribution as the whole
                dataset. Defaults to False.
            stratify_by (np.ndarray, optional): A 1D array of additional labels
                to stratify the split by. Defaults to None. If none is provided
                will use the actual labels in the dataset. This is useful if
                you want to stratify based on some other property of the data.

        Returns:
            (tuple): A train dataset with (1-split) fraction of the data and a
            validation dataset with split fraction of the data

        Note:
            Shuffling the dataset will at one point cause double the size of
            the dataset to be loaded into RAM. If this is an issue, I suggest
            you store your dataset on disk split up into validation and train
            so you don't do this splitting in memory.
        """
        train_split, val_split = self.get_split_indicies(
            split, shuffle, seed, stratified, stratify_by)
        train_data = self.__class__(
            self.x[train_split],
            y=None if not self.has_labels else self.y[train_split],
            ids=None if not self.has_ids else self.ids[train_split])
        val_data = self.__class__(
            self.x[val_split],
            y=None if not self.has_labels else self.y[val_split],
            ids=None if not self.has_ids else self.ids[val_split])
        return train_data, val_data

    def kfold(self, k=5, shuffle=False, seed=None):
        """
        An iterator that yields one fold of the kfold split of the data
        # Arguments:
            k -- The number of folds to use. Default: 5
            shuffle -- Whether or not to randomly sample the validation set
                       and train set from the parent dataset. Default: False
            seed -- A seed for the random number generator (optional).

        # Yields
            A train dataset with 1-1/k fraction of the data and a validation
            dataset with 1\k fraction of the data. Each subsequent validation
            set contains a different region of the entire dataset. The
            intersection of each validation set is empty and the union of each
            is the entire dataset.

        # Note
            Shuffling the dataset will at one point cause double the size of
            the dataset to be loaded into RAM. If this is an issue, I suggest
            you store your dataset on disk split up into validation and train
            so you don't do this splitting in memory. You can set the
            destroy_self flag to True if you can afford the split, but want to
            reclaim the memory from the parent dataset.
        """
        for train_split, val_split in self.get_kfold_indices(k, shuffle, seed):
            train_data = NpDataset(
                self.x[train_split],
                y=None if not self.has_labels else self.y[train_split],
                ids=None if not self.has_ids else self.ids[train_split])
            val_data = NpDataset(
                self.x[val_split],
                y=None if not self.has_labels else self.y[val_split],
                ids=None if not self.has_ids else self.ids[val_split])
            yield train_data, val_data


### TRAINING ###
class GeneratorEnqueuer(BatchGenerator):
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
        self.epoch_logs[metric.__name__] = metric.accumulate()

    def on_epoch_end(self):
        for metric_name, score in self.epoch_logs.items():
            self.setdefault(metric_name, []).append(score)

    def log_validation_metric(self, metric):
        self.epoch_logs["val_" + metric.__name__] = metric.accumulate()


class LossManager(object):

    @resettable
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
        self.__loss_scores[name] = self.__loss_weight_dict[name] * \
            self.__loss_dict[name](
                *[getattr(model, loss_input) for loss_input
                  in self.__loss_input_dict[name]],
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
                            loss_input=loss_input, loss_name=loss_name))

    def loss(self, model, targets):
        # This means we need to verify that the input arguments for the loss
        # exist, and notify the user if they don't
        if self.__verify_loss_args:
            self.verify_args(model)
            self.__verify_loss_args = False

        # Compute the loss
        return sum(self._compute_single_loss(model, targets, loss_name) for
                   loss_name in self.__loss_names)

    def get_loss_score(self, name=None):
        if name is None:
            assert not len(self.__loss_names), "Need to specify a loss if " \
                "using multiple losses."
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
        return {"name": name,
                "loss": loss_fn,
                "inputs": inputs,
                "weight": weight}

    def clear_losses(self):
        self.reset()


class OptimizerManager(object):

    @resettable
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
        return {"name": name,
                "optimizer": optimizer}

    def clear_optimizers(self):
        self.reset()


### REGISTRY ###
METRICS_REGISTRY = {}


def register_metric(name, metric):
    metric.__name__ = name
    METRICS_REGISTRY[name] = metric


def load_metric(name):
    return METRICS_REGISTRY[name]


### CALLBACKS ###
class CallbackList(object):
    """Container abstracting a list of callbacks.
    # Arguments
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = {} if logs is None else logs
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs=logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = {} if logs is None else logs
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, step, epoch, logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = {} if logs is None else logs
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(step, epoch, logs=logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0.
                and delta_t_median > 0.95 * self._delta_t_batch
                and delta_t_median > 0.1):
            warnings.warn('Method on_batch_begin() is slow compared '
                          'to the batch update (%f). Check your callbacks.' %
                          delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, step, epoch, logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = {} if logs is None else logs
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(step, epoch, logs=logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0.
                and (delta_t_median > 0.95 * self._delta_t_batch
                     and delta_t_median > 0.1)):
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.' %
                          delta_t_median)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = {} if logs is None else logs
        for callback in self.callbacks:
            callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        logs = {} if logs is None else logs
        for callback in self.callbacks:
            callback.on_train_end(logs=logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    """Abstract base class used to build new callbacks.
    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.
    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:
        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, step, epoch, logs=None):
        pass

    def on_batch_end(self, step, epoch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class ProgressBar(Callback):

    def __init__(self, steps, epochs=0):
        super(ProgressBar, self).__init__()
        self.steps = steps
        self.epochs = epochs
        self.last_step = 0
        self.progbar = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.epochs:
            print("Epoch {curr}/{total}".format(curr=epoch + 1,
                                                total=self.epochs))
        # Create a new progress bar for the epoch
        self.progbar = tqdm(total=self.steps)
        self.last_step = 0
        # Store the logs for updating the postfix
        self.epoch_logs = logs

    def on_batch_end(self, step, epoch, logs=None):
        self.progbar.set_postfix(self.epoch_logs)
        self.progbar.update(step - self.last_step)
        self.last_step = step

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_logs = logs
        self.progbar.set_postfix(logs)
        # 0 because we've already finished all steps
        self.progbar.update(0)
        self.progbar.close()


class ModelCheckpoint(Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        monitor_val: whether or not to monitor the validation quantity.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor, verbose=0,
                 save_best_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or 'auc' in self.monitor or 'iou' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch)
            if self.save_best_only:
                current = logs[self.monitor]
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        self.model.save_state(filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                self.model.save_state(filepath)


class Plotter(Callback):

    def __init__(self, monitor, scale='linear', plot_during_train=True, save_to_file=None, block_on_end=True):
        super().__init__()
        if plt is None:
            raise ValueError("Must be able to import Matplotlib to use the Plotter.")
        self.scale = scale
        self.monitor = monitor
        self.plot_during_train = plot_during_train
        self.save_to_file = save_to_file
        self.block_on_end = block_on_end

        if self.plot_during_train:
            plt.ion()

        self.fig = plt.figure()
        self.title = "{} per Epoch".format(self.monitor)
        self.xlabel = "Epoch"
        self.ylabel = self.monitor
        self.ax = self.fig.add_subplot(111, title=self.title,
                                       xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.set_yscale(self.scale)
        self.x = []
        self.y_train = []
        self.y_val = []

    def on_train_end(self, logs=None):
        if self.plot_during_train:
            plt.ioff()
        if self.block_on_end:
            plt.show()
        return

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.x.append(len(self.x))
        self.y_train.append(logs[self.monitor])
        self.y_val.append(logs["val_" + self.monitor])
        self.ax.clear()
        # # Set up the plot
        self.fig.suptitle(self.title)

        self.ax.set_yscale(self.scale)
        # Actually plot
        self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')
        self.fig.canvas.draw()
        # plt.pause(0.5)
        if self.save_to_file is not None:
            self.fig.savefig(self.save_to_file)
        return

class MetricLogger(Callback):

    def __init__(self, log_fname):
        super().__init__()
        self.log_fname = log_fname

    def on_epoch_end(self, epoch, train_logs=None, val_logs=None):
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        # Write the info to the log
        with open(self.log_fname, 'a') as log_file:
            print("Epoch: %s" % epoch, file=log_file)
            if len(train_logs) > 0:
                print("Train", file=log_file)
            for metric, values in train_logs.items():
                print("\t{}: {}".format(metric, values[-1]), file=log_file)
            if len(val_logs) > 0:
                print("Val", file=log_file)
            for metric, values in val_logs.items():
                print("\t{}: {}".format(metric, values[-1]), file=log_file)
            print("", file=log_file)


class ReduceLROnPlateau(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    # Example
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    # Arguments
        optimizer: the pytorch optimizer to modify
        monitor: quantity to be monitored.
        monitor_val: whether or not to monitor the validation quantity.
        factor: factor by which the learning rate will
            be reduced. new_lr = lr * factor
        patience: number of epochs with no improvement
            after which learning rate will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode,
            lr will be reduced when the quantity
            monitored has stopped decreasing; in `max`
            mode it will be reduced when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        epsilon: threshold for measuring the new optimum,
            to only focus on significant changes.
        cooldown: number of epochs to wait before resuming
            normal operation after lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self, optimizer, monitor, monitor_val=True, factor=0.1, patience=10,
                 verbose=0, mode='auto', epsilon=1e-4, cooldown=0, min_lr=0):
        super(ReduceLROnPlateau, self).__init__()

        self.optimizer = optimizer
        self.monitor = monitor
        self.monitor_val = monitor_val
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau '
                             'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            warnings.warn('Learning Rate Plateau Reducing mode %s is unknown, '
                          'fallback to auto mode.' % (self.mode),
                          RuntimeWarning)
            self.mode = 'auto'
        if (self.mode == 'min' or
           (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None, **kwargs):
        self._reset()

    def on_epoch_end(self, epoch, train_logs=None, val_logs=None):
        logs = val_logs if self.monitor_val else train_logs
        logs = logs or {}

        current = logs.get(self.monitor)[-1]
        if current is None:
            warnings.warn(
                'Reduce LR on plateau conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                if self.wait >= self.patience:
                    reduced_lr = False
                    for param_group in self.optimizer.param_groups:
                        old_lr = param_group['lr']
                        if old_lr > self.min_lr:
                            param_group['lr'] = max(old_lr * self.factor, self.min_lr)
                            reduced_lr = True
                    if reduced_lr:
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning rate by %s factor.' % (
                                  epoch + 1, self.factor))

                self.wait += 1

    def in_cooldown(self):
        return self.cooldown_counter > 0


class LRScheduler(Callback):

    def __init__(self, optimizer, schedule, verbose=0):
        super().__init__()
        self.optimizer = optimizer
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, train_logs=None, val_logs=None):
        new_lr = self.schedule(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        if self.verbose > 0:
            print('\nEpoch %05d: LRScheduler setting lr to %s.' % (epoch + 1, new_lr))


"""
METRICS
Files Included:
    __init__.py
    abstract_metrics.py
    accuracy_metrics.py
Missing Files:
    segmentation_metrics.py
"""


class Metric(object):
    """
    The abstract metric that defines the metric API. Some notes on it:

    - Passing a function of the form `metric(y_pred, y_true)` to an abstract
    metric will use that function to calculate the score on a batch.

    - The accumulate method is called at the end of each batch to calculate the
    aggregate score over the entire epoch thus far.
        - See `AverageMetric` for an example what an accumulate method might
        look like.

    - The reset method is called at the end of each epoch or validation run. It
    simply overwrites the attributes of the metric with its attributes at
    initialization.

    Metrics are callable like any fuction and take as input:
    ```
    batch_score = metric(y_pred, y_true)
    ```
    where `y_true` are the labels for the batch and `y_pred` are the
    predictions

    To implement your own custom metric, override the `score` function and
    the `accumulate` function. If you just want to average the scores over
    the epoch, consider using `AverageMetric` and just overriding the `score`
    function.
    """
    def __init__(self, metric_func=None):
        self.metric_func = metric_func
        self.__name__ = self.__class__.__name__.lower() \
            if metric_func is None else metric_func.__name__
        self.__original_dict__ = None

    def __call__(self, y_pred, y_true):
        """
        Makes the metric a callable function. Used by some metrics to perform
        some overhead work like checking validity of the input, or storing
        values like batch size or input shape.
        """
        # Save the original dict on the first call
        if self.__original_dict__ is None:
            self.__original_dict__ = copy.deepcopy(self.__dict__)
        # Default metric will just score the predictions
        return self.score(y_pred, y_true)

    def score(self, y_pred, y_true):
        """
        Calculates the metric score over a batch of labels and predictions.

        Args:
            y_pred: The predictions for the batch
            y_true: The labels for the batch

        Returns:
            The metric score calculated over the batch input as a scalar
            torch tensor.
        """
        if self.metric_func is not None:
            return self.metric_func(y_pred, y_true)
        else:
            raise NotImplementedError()

    def accumulate(self):
        """
        """
        raise NotImplementedError()

    def reset(self):
        if self.__original_dict__ is not None:
            self.__dict__ = copy.deepcopy(self.__original_dict__)
        return self


class AverageMetric(Metric):
    """
    An abstract metric that accumulates the batch values from the metric
    by averaging them together. If any function is input into the fit
    function as a metric, it will automatically be considered an AverageMetric.
    """
    def __init__(self, metric_func=None):
        super(AverageMetric, self).__init__(metric_func=metric_func)
        self.metric_sum = 0.
        self.sample_count = 0

    def __call__(self, y_pred, y_true):
        assert y_true.size(0) == y_pred.size(0), "Batch Size of labels and" \
            "predictions must match for AverageMetric."
        score = super(AverageMetric, self).__call__(y_pred, y_true)
        self.sample_count += y_pred.size(0)
        self.metric_sum += (score.item() * y_pred.size(0))
        return score

    def accumulate(self):
        return self.metric_sum / self.sample_count


class Accuracy(AverageMetric):
    """
    Computes the accuracy over predictions

    Args:
        None

    Returns:
        An accuracy metric that can maintain its own internal state.

    Inputs:
        y_pred (torch.FloatTensor): A 2D Float Tensor with the predicted
            probabilites for each class.
        y_true (torch.LongTensor): A 1D torch LongTensor of the correct classes
    Outputs:
        A scalar tensor equal to the accuracy of the y_pred

    """
    def score(self, y_pred, y_true):
        # Expect output and target to be B x 1 or B x C or target can be
        # B (with ints from 0 to C-1)
        assert y_pred.dim() == 2, "y_pred should be a 2-dimensional tensor"
        total = y_true.size(0)
        # Turn it into a 1d class encoding
        if y_true.dim() == 2:
            if y_true.size(1) > 1:
                raise NotImplementedError(
                    "Multiclass with 2d targets is not impelemented yet")
            y_true = y_true.squeeze(1).long()

        # Change the y_pred to have two cols if it only has 1
        if y_pred.size(1) == 1:
            # Want to consider the 0.5 case
            y_pred = (y_pred >= 0.5).float()
            y_pred = torch.cat([1 - y_pred, y_pred], dim=1)

        # Compute the accuracy
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y_true).float().sum(0)

        return (correct / total) * 100.


class AccuracyWithLogits(Accuracy):

    """An accuracy metric that takes as input the logits. See `Accuracy` for
    more details.
    """
    def score(self, y_pred, y_true):
        if y_pred.dim() == 2 and y_pred.size(1) > 1:
            y_pred = F.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
        return super().score(y_pred, y_true)


class TopKAccuracy(Accuracy):
    """Computes the precision@k for the specified values of k

    Args:
        k (int): The k to calculate precision@k (topk accuracy)

    Returns:
        A TopKAccuracy metric that can maintain its own internal state.

    Inputs:
        y_pred (torch.FloatTensor) A 2D Float Tensor with the predicted
            probabilites for each class.
        y_true (torch.LongTensor) A 1D torch LongTensor of the correct classes

    Outputs:
        A scalar tensor equal to the topk accuracy of the y_pred

    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def score(self, y_pred, y_true):
        assert y_true.dim() == y_pred.dim() - 1 == 1
        channel_dim = 1
        batch_size = y_true.size(0)
        # Check to see if there's only 1 channel (then its binary
        # classification)
        if y_pred.size(channel_dim) == 1:
            y_pred = y_pred.squeeze(channel_dim)  # B x ...
            # 2 x B x ... -> B x 2 x ...
            y_pred = y_pred.stack([1 - y_pred, y_pred]).t()

        # Get the indicies of the topk along the channel dim
        _, pred = y_pred.topk(self.k, channel_dim, True, True)  # B x k x ...
        pred = pred.t()  # k x B x ...
        # target: B -> 1 x B -> k x B x ...
        correct = pred.eq(y_true.view(1, -1).expand_as(pred))

        correct_k = correct[:self.k].view(-1).float().sum(0)

        # Accumulate results
        return 100. * correct_k / batch_size


accuracy = Accuracy()
accuracy_with_logits = AccuracyWithLogits()
top2_accuracy = TopKAccuracy(2)
top3_accuracy = TopKAccuracy(3)
top5_accuracy = TopKAccuracy(5)

register_metric('accuracy', accuracy)
register_metric('accuracy_with_logits', accuracy_with_logits)
register_metric('top2_accuracy', top2_accuracy)
register_metric('top3_accuracy', top3_accuracy)
register_metric('top5_accuracy', top5_accuracy)


"""
MODELS
"""

python_iterables = {list, set, tuple, frozenset}


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def standardize_list_input(inputs):
    if type(inputs) in python_iterables:
        return list(inputs)
    return [inputs]


def standardize_metric_input(metrics):
    old_metrics = standardize_list_input(metrics)
    metrics = []
    for metric in old_metrics:
        if isinstance(metric, str):
            metrics.append(load_metric(metric))
        elif isinstance(metric, Metric):
            metrics.append(metric)
        else:
            metrics.append(AverageMetric(metric))
    return metrics


# TODO Not sure whether I'll need to seperate RL models and SL models.
# Hopefully I planned this out right
class SLModel(nn.Module):
    def __init__(self, torch_module=None):
        super(SLModel, self).__init__()
        self.to_cuda = use_cuda
        self.loss_in = []
        self.torch_module = torch_module

        self.loss_manager = LossManager()
        self.optimizer_manager = OptimizerManager()

    def infer_inputs(self, *inputs, **kwargs):
        self.cast_model_to_cuda()
        with torch.no_grad():
            self.forward(*inputs, **kwargs)

    def parameters(self, *args, **kwargs):
        params = super(SLModel, self).parameters(*args, **kwargs)
        param_peek = peek(params)
        if param_peek is None:
            warnings.warn("Model has no parameters! Did you forget to call "
                          "infer_inputs?")
            return []
        return param_peek[1]

    def forward(self, *inputs, **kwargs):
        if self.torch_module is not None:
            self.loss_in = self.torch_module.forward(*inputs, **kwargs)
            return self.loss_in
        raise NotImplementedError()

    def cast_input_to_torch(self, x):
        return Variable(from_numpy(x))

    def cast_target_to_torch(self, y):
        return Variable(from_numpy(y))

    def cast_output_to_numpy(self, preds):
        return preds.data.cpu().numpy()

    def cast_model_to_cuda(self):
        if self.to_cuda:
            self.cuda()
            self.to_cuda = False
        return

    def add_optimizer(self, optimizer, name=None):
        self.optimizer_manager.add_optimizer(optimizer, name=name)

    def remove_optimizer(self, name=None):
        return self.optimizer_manager.remove_optimizer(name=name)

    def clear_optimizers(self):
        self.optimizer_manager.clear_optimizers()

    def loss(self, targets):
        return self.loss_manager.loss(self, targets)

    def add_loss(self, loss_fn, inputs=(), weight=1.0, name=None):
        inputs = standardize_list_input(inputs)
        # Use 'loss_in' if no inputs provided
        if not len(inputs):
            inputs = ['loss_in']
        return self.loss_manager.add_loss(loss_fn,
                                          inputs,
                                          weight=weight,
                                          name=name)

    def remove_loss(self, name=None):
        return self.loss_manager.remove_loss(name=name)

    def clear_losses(self):
        self.loss_manager.clear_losses()

    def compile_loss(self, loss_fn=None):
        """
        This is a function to standardize loss input and hack it to behave like
        a metric. A few key notes to remember:
            - If the loss_fn is None, it will just use the loss method
              defined by the model. This by default comes from the loss manager
              which is modified by the add_loss, remove_loss, and clear_losses
              methods. If a loss_fn is provided, then this method will clear
              all current losses from the loss manager and add the input loss
              function to it, taking as input the default "loss_in" parameter.
              If you override the model's loss function, then passing a loss_fn
              will have no effect!
            - If there is more than one loss in the loss manager, then this
              function will also return metric versions of all the auxilary
              losses. The overall loss function is only computed once,
              the auxilary loss scores are taken from loss cache.

        Args:
            loss_fn: The loss function to compile. Defaults to None. See above
                note for explanation of behavior when None and when not None.

        Returns:
            (tuple): All the relevant loss functions in a tuple. See above note
                for more explanation about how this return value is determined
        """
        # if loss_fn is defined, clear the losses, and set it to the input
        # loss_fn
        if loss_fn is not None:
            if len(self.loss_manager):
                warnings.warn("Loss manager is not empty, but loss_fn passed "
                              "passed to fit_generator or validate_generator."
                              " Clearing all past losses.")
                self.clear_losses()
                self.add_loss(loss_fn)

        # Compile the main loss
        def loss(preds, targets):
            # Preds are not used, just works as metric)
            return self.loss(targets)

        # Compile the auxilary losses, the main loss must be called before
        # the auxilary losses
        aux_losses = []
        # Only account for auxilary losses if there is more than one loss
        if len(self.loss_manager) > 1:
            for name in self.loss_manager.names:
                # Using the default value gets around the problem of late
                # binding.
                # https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
                def aux_loss(preds, targets, name=name):
                    # Preds are not used, just hack to make it behave like
                    # metric
                    return self.loss_manager.get_loss_score(name=name)
                metric_aux_loss = AverageMetric(aux_loss)
                # Change the name for logging
                metric_aux_loss.__name__ = name
                aux_losses.append(metric_aux_loss)

        return (AverageMetric(loss), *aux_losses)

    def train_on_batch(self, x, target, optimizers, loss_fn, metrics=()):
        """
        Trains the SLModel on a single batch of data.

        Args:
            x: A batch of input into the model.
            target: The corresponding labels for the batch x.
            optimizers: A list of optimizers to run with the model.
            loss_fn: The loss function to run on the model
            metrics: A list of metrics to calculate on the output of the model

        Returns:
            A tuple where the first element is the loss over the batch and the
            second element is a list of the scores corresponding to the input
            metrics.
        """
        self.cast_model_to_cuda()
        self.train()
        # Cast inputs to a torch variable
        torch_x = self.cast_input_to_torch(x)
        torch_target = self.cast_target_to_torch(target)
        # Make the prediction
        torch_preds = self(torch_x)
        # Calculate the loss
        loss = loss_fn(torch_preds, torch_target)
        # Update the weights
        [optimizer.zero_grad() for optimizer in optimizers]
        loss.backward()
        [optimizer.step() for optimizer in optimizers]
        # Calculate the metrics
        metric_scores = [
            metric(torch_preds, torch_target) for metric in metrics
        ]
        # Clean up some variables
        self.zero_grad()
        del torch_x
        del torch_target
        del torch_preds
        if use_cuda:
            torch.cuda.empty_cache()
        return loss, metric_scores

    def validate_on_batch(self, x, target, metrics):
        self.cast_model_to_cuda()
        self.eval()
        with torch.no_grad():
            # Cast inputs to a torch variable and set to volatile for inference
            torch_x = self.cast_input_to_torch(x)
            torch_target = self.cast_target_to_torch(target)
            # Make the prediction
            torch_preds = self(torch_x)
            preds = self.cast_output_to_numpy(torch_preds)
            # Calculate the metrics
            metric_scores = [
                metric(torch_preds, torch_target) for metric in metrics
            ]
            # Clean up some variables
            del torch_x
            del torch_preds
            del torch_target
            if use_cuda:
                torch.cuda.empty_cache()
        return metric_scores, preds

    def validate_generator(self,
                           val_generator,
                           validation_steps,
                           loss_fn=None,
                           metrics=(),
                           verbose=0):
        self.cast_model_to_cuda()
        metrics = standardize_metric_input(metrics)
        if loss_fn is not None or len(self.loss_manager):
            loss_fn, *aux_loss_fns = self.compile_loss(loss_fn)
            metrics = [loss_fn] + metrics + aux_loss_fns
        # Set up the logs
        logs = TrainingLogs()
        # Set the model to eval mode
        self.eval()
        callbacks = [ProgressBar(validation_steps)] if verbose > 0 else []
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin(logs=logs)
        callbacks.on_epoch_begin(0, logs=logs.epoch_logs)
        for step in range(validation_steps):
            callbacks.on_batch_begin(epoch=0, step=step, logs=logs.batch_logs)
            x, target = next(val_generator)
            b_metrics, _ = self.validate_on_batch(x, target, metrics)
            for metric, score in zip(metrics, b_metrics):
                logs.log_metric(metric, score)
            callbacks.on_batch_end(epoch=0, step=step, logs=logs.batch_logs)
        callbacks.on_epoch_end(0, logs=logs.epoch_logs)
        callbacks.on_train_end(logs=logs)
        return logs.epoch_logs

    def fit_generator(self,
                      generator,
                      steps_per_epoch,
                      epochs,
                      validation_data=None,
                      validation_steps=0,
                      metrics=(),
                      callbacks=(),
                      initial_epoch=0,
                      verbose=1):
        self.cast_model_to_cuda()
        # Standardize the input
        optimizers = self.optimizer_manager.optimizers
        loss_fn, *aux_loss_fns = self.compile_loss()
        metrics = standardize_metric_input(metrics) + aux_loss_fns
        callbacks = CallbackList(callbacks)
        # If the verbosity is set, set up the progress bar
        if verbose > 0:
            callbacks.append(ProgressBar(steps_per_epoch, epochs=epochs))
        # Register the model with each callback
        callbacks.set_model(self)
        # Save whether we will need to run validation
        run_validation = (validation_steps >
                          0) and validation_data is not None
        logs = TrainingLogs()

        # Run the callbacks
        callbacks.on_train_begin(logs=logs)
        # Loop through all the epochs
        for epoch in range(initial_epoch, epochs):
            # Put the model in train mode
            self.train()
            # Reset the metrics
            loss_fn = loss_fn.reset()
            metrics = [metric.reset() for metric in metrics]
            # Run the callbacks
            logs.on_epoch_begin()
            callbacks.on_epoch_begin(epoch, logs=logs.epoch_logs)
            # Run each step of the epoch with a progress bar
            for step in range(steps_per_epoch):
                # Run the callbacks
                callbacks.on_batch_begin(
                    epoch=epoch, step=step, logs=logs.batch_logs)
                x, target = next(generator)
                b_loss, b_metrics = self.train_on_batch(
                    x, target, optimizers, loss_fn, metrics)
                # Add stats to the logs
                logs.log_metric(loss_fn, b_loss)
                for score, metric in zip(b_metrics, metrics):
                    logs.log_metric(metric, score)
                # Run the callbacks
                callbacks.on_batch_end(
                    epoch=epoch, step=step, logs=logs.batch_logs)

            # Check if we need to run validation
            if run_validation:
                loss_fn = loss_fn.reset()
                metrics = [metric.reset() for metric in metrics]
                self.validate_generator(
                    validation_data,
                    validation_steps,
                    metrics=([loss_fn] + metrics))
                # Log the loss and metrics
                for metric in [loss_fn] + metrics:
                    logs.log_validation_metric(metric)
            # Run the callbacks
            logs.on_epoch_end()
            callbacks.on_epoch_end(epoch, logs=logs.epoch_logs)
        # Run the callbacks
        callbacks.on_train_end(logs=logs)
        # Put the model back in eval mode
        self.eval()
        return logs

    def predict_on_batch(self, x):
        self.cast_model_to_cuda()
        self.eval()
        with torch.no_grad():
            # Cast inputs to a torch variable and set to volatile for inference
            torch_x = self.cast_input_to_torch(x)
            # Make the prediction
            torch_preds = self(torch_x)
            preds = self.cast_output_to_numpy(torch_preds)
            self.zero_grad()
            del torch_x
            del torch_preds
            if use_cuda:
                torch.cuda.empty_cache()
        # cast to numpy and return
        return preds

    def predict_generator(self, generator, prediction_steps, verbose=0):
        self.cast_model_to_cuda()
        self.eval()
        preds = []
        # Loop through all the steps
        progbar = tqdm if verbose > 0 else lambda x: x
        for _ in progbar(range(prediction_steps)):
            x = next(generator)
            batch_preds = self.predict_on_batch(x)
            # Check to make sure the ndim is the same
            if len(preds) > 0:
                assert batch_preds.ndim == preds[-1].ndim
            preds.append(batch_preds)

        # Supports variable sized predictions - get the biggest possible shape
        num_preds = sum(len(batch_preds) for batch_preds in preds)
        max_shape = [num_preds] + [
            max(preds[n].shape[i] for n in range(len(preds)))
            for i in range(1, preds[0].ndim)
        ]
        full_preds = np.zeros(max_shape, dtype=preds[0].dtype)
        # Fill in the predictions array
        cur_pred_ind = 0
        for batch_preds in preds:
            preds_slice = (slice(cur_pred_ind,
                                 cur_pred_ind + len(batch_preds)), ) + tuple(
                                     slice(batch_preds.shape[i])
                                     for i in range(1, batch_preds.ndim))
            full_preds[preds_slice] = batch_preds
            cur_pred_ind += len(batch_preds)

        return full_preds

    def save_state(self, save_path):
        return torch.save(self.state_dict(), save_path)

    def load_state(self, load_path):
        self.load_state_dict(torch.load(load_path))


"""
LAYERS
Files Included:
    layer_utils.py
    layer.py
    functions.py
    core.py
    merge.py
    noise.py
    pooling.py
    convolutional.py
    recurrent.py
    attentional.py
"""
# layer_utils.py
pool_types = {"no_pool": lambda *args, **kwargs: lambda x: x,
              "max": nn.MaxPool1d,
              "avg": nn.AvgPool1d}
activation_types = {name.lower(): cls for name, cls in nn.modules.activation.__dict__.items() if isinstance(cls, type)}
activation_types["linear"] = None


def get_type(item_type, type_dict, fail_message):
    try:
        if not isinstance(item_type, str):
            return item_type
        return type_dict[item_type]
    except KeyError:
        raise NotImplementedError(fail_message)


def get_pool_type(pool_type):
    return get_type(pool_type, pool_types, "pool type %s" % pool_type)


def get_activation_type(activation_type):
    return get_type(activation_type, activation_types, "Activation %s" % activation_type)


def construct_n_layers(layer_factory, num_layers, input_size, output_size, *args, **kwargs):
    layers = nn.ModuleList([layer_factory(input_size, output_size, *args, **kwargs)])
    for _ in range(num_layers - 1):
        layers.append(layer_factory(output_size, output_size, *args, **kwargs))
    return layers


def get_input_shape(inputs):
    return tuple(inputs.size())[1:]


def builder(func):

    def build_layer(self, *args, **kwargs):
        assert not self.built, "Cannot build a layer multiple times!"
        func(self, *args, **kwargs)
        if use_cuda:
            self.cuda()
        self.built = True

    return build_layer


# layer.py
class Layer(nn.Module):

    def __init__(self):
        super(Layer, self).__init__()
        self.built = False

    def forward(self, *input):
        raise NotImplementedError()

    def reset_parameters(self):
        pass

    def __str__(self):
        return self.__class__.__name__ + "()"

    def __repr__(self):
        return str(self)

# functions.py

# TODO: Add a cropping function


def pad_tensor(tensor, length, pad_value=0.0, dim=0):
    # tensor is Li x E
    tensor = tensor.transpose(0, dim).contiguous()
    if tensor.size(0) == length:
        tensor = tensor
    elif tensor.size(0) > length:
        return tensor[:length]
    else:
        tensor = torch.cat([tensor, Variable(zeros(length - tensor.size(0), *tensor.size()[1:]).fill_(pad_value),
                                             requires_grad=False)])
    return tensor.transpose(0, dim).contiguous()


def pad_sequences(tensors, pad_value=0.0, length_last=False):
    # tensors is B x Li x E
    # First find how long we need to pad until
    length_dim = -1 if length_last else 0
    assert len(tensors) > 0
    if length_last:
        assert all(tuple(seq.size())[:-1] == tuple(tensors[0].size())[:-1] for seq in tensors)
    else:
        assert all(tuple(seq.size())[1:] == tuple(tensors[0].size())[1:] for seq in tensors)
    seq_lens = [seq.size(length_dim) for seq in tensors]
    max_len = max(seq_lens)
    # Out is B x L x E
    # print([tuple(pad_tensor(tensors[i], max_len).size()) for i in range(len(tensors))])
    if length_last:
        return torch.stack(
            [pad_tensor(tensors[i].transpose(0, length_dim), max_len, pad_value=pad_value).transpose(0, length_dim)
             for i in range(len(tensors))]), seq_lens

    return torch.stack([pad_tensor(tensors[i], max_len, pad_value=pad_value) for i in range(len(tensors))]), seq_lens


def unpad_sequences(padded_tensors, seq_lens, length_last=False):
    length_dim = -1 if length_last else 0
    if length_last:
        return [padded_tensor.transpose(0, length_dim)[:seq_len].transpose(0, length_dim) for padded_tensor, seq_len in
                zip(padded_tensors, seq_lens)]
    return [padded_tensor[:seq_len] for padded_tensor, seq_len in zip(padded_tensors, seq_lens)]


def pack_sequences(tensors):
    # tensors is B x Li x E
    assert len(tensors) > 0
    assert all(seq.size(1) == tensors[0].size(1) for seq in tensors)
    seq_lens = [seq.size(0) for seq in tensors]
    return torch.cat(tensors), seq_lens


def unpack_sequences(packed_tensors, seq_lens):
    # Find the start inds of all of the sequences
    seq_starts = [0 for _ in range(len(seq_lens))]
    seq_starts[1:] = [seq_starts[i-1] + seq_lens[i-1] for i in range(1, len(seq_starts))]
    # Unpack the tensors
    return [packed_tensors[seq_starts[i]:seq_starts[i] + seq_lens[i]] for i in range(len(seq_lens))]


def kmax_pooling(x, dim, k):
    index = x.topk(min(x.size(dim), k), dim=dim)[1].sort(dim=dim)[0]
    x = x.gather(dim, index)
    if x.size(dim) < k:
        x = pad_tensor(x, k, dim=dim)
    return x


def pad_numpy_to_length(x, length):
    if len(x) < length:
        return np.concatenate([x, np.zeros((length - len(x),) + x.shape[1:])], axis=0)
    return x


def pad_numpy_to_shape(x, shape):
    pad_diffs = [length - x_len for x_len, length in zip(x.shape, shape)]
    pad_args = [(0, pad_diff) for pad_diff in pad_diffs] + [(0, 0)] * (x.ndim - len(shape))
    return np.pad(x, pad_args, mode='constant')


def create2d_mask(x, seq_lens):
    # seq_lens are of shape B x 2
    # x is of shape B x H x W x F

    # shape is B x H x 1 x 1
    seq_lens_heights = seq_lens.view(-1, 2, 1, 1)[:, 0:1]
    seq_lens_widths = seq_lens.view(-1, 1, 2, 1)[:, :, 1:2]
    mask_height = Variable((arange(x.size(1)).long().view(1, -1, 1, 1) >= seq_lens_heights),
                           requires_grad=False)
    # shape is B x 1 x W x 1
    mask_width = Variable((arange(x.size(2)).long().view(1, 1, -1, 1) >= seq_lens_widths),
                           requires_grad=False)
    # shape is B x H x W x 1
    mask = mask_height | mask_width
    return mask


def seq_softmax(x, return_padded=False):
    # x comes in as B x Li x F, we compute the softmax over Li for each F
    x, lens = pad_sequences(x, pad_value=-float('inf'))  # B x L x F
    shape = tuple(x.size())
    assert len(shape) == 3
    x = F.softmax(x, dim=1)
    assert tuple(x.size()) == shape
    if return_padded:
        return x, lens
    # Un-pad the tensor and return
    return unpad_sequences(x, lens)  # B x Li x F



# core.py

# TODO Create abstract layers for layers with params that includes weight regularizers


def Input(*input_shape):
    # Use 1 for the batch size
    return zeros(1, *input_shape)

def build_fully_connected(units, input_shape, use_bias=True, \
                          activation='linear', num_layers=1, batchnorm=False,
                          input_dropout=0.0, dropout=0.0):
    assert len(input_shape) == 1, "Input to FullyConnected layer " \
        "can only have 1 dimension. {} has {} dimensions" \
        "".format(input_shape, len(input_shape))
    input_size, output_size = input_shape[0], units
    layer = nn.Sequential()
    if input_dropout:
        layer.add_module(name="input-dropout", module=nn.Dropout(input_dropout))
    for i in range(num_layers):
        layer_input = input_size if i == 0 else output_size
        layer.add_module(name="fullyconnected-%s" % i, module=nn.Linear(layer_input, output_size, bias=use_bias))
        if activation != "linear":
            layer.add_module(name="{}-{}".format(activation, i), module=get_activation_type(activation)())
        if batchnorm:
            layer.add_module(name="batchnorm-%s" % i, module=nn.BatchNorm1d(output_size))
        if dropout:
            layer.add_module(name="dropout-%s" % i, module=nn.Dropout(dropout))
    logging.info("Creating layer: %r" % layer)
    return layer


class FullyConnected(Layer):
    """Just your regular fully-connected NN layer.
        `FullyConnected` implements the operation:
        `output = activation(dot(input, kernel) + bias)`
        where `activation` is the element-wise activation function
        passed as the `activation` argument, `kernel` is a weights matrix
        created by the layer, and `bias` is a bias vector created by the layer
        (only applicable if `use_bias` is `True`).
        Note: if the input to the layer has a rank greater than 2, then
        it is flattened prior to the initial dot product with `kernel`.
        # Example
        ```python
            # A layer that takes as input tensors of shape (*, 128)
            # and outputs arrays of shape (*, 64)
            layer = FullyConnected(128, 64)
            tensor = torch.randn(32, 128)
            output = layer(tensor)
        ```
        # Arguments
            input_size: Positive integer, dimensionality of the input space.
            output_size: Positive integer, dimensionality of the input space.
            activation: String, Name of activation function to use
                (supports "tanh", "relu", and "linear").
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: `a(x) = x`).
            use_bias: Boolean, whether the layer uses a bias vector.
        # Input shape
            2D tensor with shape: `(batch_size, input_size)`.
        # Output shape
            2D tensor with shape: `(batch_size, output_size)`.
        """

    def __init__(self, units, input_shape=None,
                 use_bias=True, activation='linear', num_layers=1,
                 batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(FullyConnected, self).__init__()
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.batchnorm = batchnorm
        self.input_dropout = input_dropout
        self.dropout = dropout

        # We'll initialize the layers in the first forward call
        self.layers = []

    @builder
    def __build_layer(self, inputs):
        if self.input_shape is None:
            self.input_shape = get_input_shape(inputs)
        self.layers = build_fully_connected(
            self.units, self.input_shape, use_bias=self.use_bias,
            activation=self.activation, num_layers=self.num_layers,
            batchnorm=self.batchnorm, input_dropout=self.input_dropout,
            dropout=self.dropout
        )

    def forward(self, inputs):
        if not self.built:
            self.__build_layer(inputs)
        return self.layers(inputs)

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.Linear):
                logging.info("Resetting layer %s" % layer)
                layer.reset_parameters()

    def __str__(self):
        return "%r" % self.layers


class Flatten(Layer):
    """Flattens the input. Does not affect the batch size.
        # Example
        ```python
            flatten = Flatten()
            tensor = torch.randn(32, 2, 3)
            # The output will be of shape (32, 6)
            output = flatten(tensor)
        ```
        """

    def __init__(self):
        super(Flatten, self).__init__()

    def __str__(self):
        return "Flatten"

    def forward(self, x):
        return flatten(x)


class Lambda(Layer):
    """Wraps arbitrary expression as a `Module` object. The input function must
    have a self argument first!
    # Examples

   ```python
        # add a x -> x^2 layer
        layer = Lambda(lambda self, x: x ** 2))
    ```
    ```python
        # add a layer that returns the concatenation
        # of the positive part of the input and
        # the opposite of the negative part
        def antirectifier(self, x):
            x = self.fc(x)
            x -= torch.mean(x, dim=1, keepdim=True)
            pos = F.relu(x)
            neg = F.relu(-x)
            return torch.cat([pos, neg], dim=1)

        layer = Lambda(antirectifier, fc=Linear(256, 128))
    ```

    # Arguments
        forward: The function to be evaluated. Should take self (the lambda object) as first argument
        layers: optional dictionary of keyword arguments that map layer names to already initialized layers.
          These layers will be accessible in the forward function by using 'self.[LAYER_NAME]', replacing
          [LAYER_NAME] for whatever the name of the layer you want to access is.
    """
    def __init__(self, forward, **layers):
        super(Lambda, self).__init__()
        for layer_name in layers:
            setattr(self, layer_name, layers[layer_name])
        self.layer_names = list(layers.keys())
        self.forward_func = forward
        self.string = "Lambda: [" + " ".join("%r" % getattr(self, layer_name) for layer_name in self.layer_names) + "]"

    def __str__(self):
        return self.string

    def forward(self, *args, **kwargs):
        return self.forward_func(self, *args, **kwargs)

    def reset_parameters(self):
        for layer_name in self.layer_names:
            getattr(self, layer_name).reset_parameters()


class MaskedInput(Layer):
    """
    A layer that takes in sequences of variable length as inputs that have
    been padded. This layer will take as input a padded torch tensor where the sequence
    length varies along the first dimension of each sample as well as a LongTensor of lengths of
    each sequence in the batch. The layer will mask the padded regions of the output of the layer
    to cut the gradient.

    # Arguments
        mask_value: The value to mask the padded input with. If passed "min" instead of a value, this will
          mask to whatever the smallest value in the batch is minus 1 (usefuly if passing to a max pooling layer).
          This defaults to 0.
    """

    def __init__(self, mask_value=0.):
        super(MaskedInput, self).__init__()
        if mask_value == 'min':
            self.mask_value_factory = lambda x: torch.min(x.data) - 1.
        else:
            self.mask_value_factory = lambda x: mask_value
        self.mask_value = mask_value
        self.__descriptor = self.__class__.__name__ + "(mask_value=%s)" % self.mask_value
        logging.info("Creating layer: %s" % self.__descriptor)

    def forward(self, x, seq_lens):
        mask = Variable((arange(x.size(1)).long().view(1, -1, 1) >= seq_lens.view(-1, 1, 1)), requires_grad=False)
        mask_value = self.mask_value_factory(x)
        return x.masked_fill(mask, mask_value)

    def __str__(self):
        return self.__descriptor


class MaskedInput2D(MaskedInput):
    """
    A layer that takes in sequences of variable length as inputs that have
    been padded. This layer will take as input a padded torch tensor where the sequence
    length varies along the first dimension of each sample as well as a LongTensor of lengths of
    each sequence in the batch. The layer will mask the padded regions of the output of the layer
    to cut the gradient.

    # Arguments
        mask_value: The value to mask the padded input with. If passed "min" instead of a value, this will
          mask to whatever the smallest value in the batch is minus 1 (usefuly if passing to a max pooling layer).
          This defaults to 0.
    """
    def forward(self, x, seq_lens):
        # seq_lens are of shape B x 2
        # x is of shape B x H x W x F
        mask = create2d_mask(x, seq_lens)
        mask_value = self.mask_value_factory(x)
        return x.masked_fill(mask, mask_value)

    def __str__(self):
        return self.__descriptor


# merge.py
class Concatenate(Layer):
    """Layer that concatenates a list of inputs.
    It takes as input a list of tensors,
    all of the same shape except for the concatenation dim, the
    dimension over which to concatenate,
    and returns a single tensor, the concatenation of all inputs.
    """
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, seq, dim=-1):
        if dim >= 0:
            dim += 1
        return torch.cat(seq, dim=dim)


class Add(Layer):
    """Layer that adds a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    # Examples
    ```python
        import pyjet
        input1 = torch.randn(32, 16)
        x1 = pyjet.layers.FullyConnected(8, activation='relu')(input1)
        input2 = torch.randn(32, 16)
        x2= pyjet.layers.FullyConnected(8, activation='relu')(input2)
        added = pyjet.layers.Add()([x1, x2])  # equivalent to added = x1 + x2
        out = pyjet.layers.FullyConnected(4)(added)
        ```
    """
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, seq):
        return sum(seq)

# noise.py
class GaussianNoise1D(Layer):

    def __init__(self, std=0.05, augment_prob=1.0):
        super().__init__()
        self.std = std
        self.augment_prob = augment_prob
        self.noise_size = tuple()
        self.noise = None
        self.mask_sample = None
        self.__descriptor = "{name}(std={std}, augment_prob={augment_prob})".format(name=self.__class__.__name__, std=std, augment_prob=augment_prob)
        logging.info("Creating layer %r" % self)

    def forward(self, x):
        if not self.training:
            return x
        self.init_noise(x)

        if self.augment_prob != 1.0:
            # 0 out the elements we don't want to change
            self.noise.data.masked_fill_(self.mask_sample > self.augment_prob, 0.)

        return x + self.noise

    def init_noise(self, x):
        # Create the noise (w/ mem optimization)
        x_shape = tuple(x.size())
        if self.noise_size != x_shape:
            self.noise = Variable(zeros(*x_shape), requires_grad=False)
            self.mask_sample = None if self.augment_prob == 1.0 else rand(*x_shape[:-1]).unsqueeze(-1)
            self.noise_size = x_shape
        else:
            self.mask_sample.uniform_()
        self.noise.data.normal_(0, std=self.std)

    def __str__(self):
        return self.__descriptor


# pooling.py
def build_strided_pool(name, kernel_size, stride=None, padding=1, dilation=1):

    layer = StridedPool.pool_funcs[name](kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    logging.info("Creating layer: %r" % layer)
    return layer


class UpSampling(Layer):

    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(UpSampling, self).__init__()
        self.upsampling = partial(F.interpolate, size=size, scale_factor=scale_factor, mode=mode)
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def calc_output_size(self, input_size):
        if self.size is not None:
            return LongTensor(self.size)
        else:
            return input_size * self.scale_factor

    def calc_input_size(self, output_size):
        if self.size is not None:
            raise ValueError("Cannot know input size if deterministic output size is used")
        else:
            return output_size / self.scale_factor

    def forward(self, x):
        # Expect x as BatchSize x Length1 x ... x LengthN x Filters
        if channels_mode == "channels_last":
            return self.unfix_input(self.upsampling(self.fix_input(x)))
        else:
            return self.upsampling(x)

    def fix_input(self, x):
        raise NotImplementedError()

    def unfix_input(self, x):
        raise NotImplementedError()

    def __str__(self):
        return "%r" % self.upsampling


class UpSampling2D(UpSampling):

    def fix_input(self, inputs):
        return inputs.permute(0, 3, 1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.permute(0, 2, 3, 1).contiguous()


class StridedPool(Layer):

    pool_funcs = {"max1d": nn.MaxPool1d,
                  "max2d": nn.MaxPool2d,
                  "max3d": nn.MaxPool3d,
                  "avg1d": nn.AvgPool1d,
                  "avg2d": nn.AvgPool2d,
                  "avg3d": nn.AvgPool3d}

    def __init__(self, pool_type, kernel_size, stride=None, padding='same', dilation=1):
        super(StridedPool, self).__init__()
        padding = (kernel_size - 1) // 2 if padding == 'same' else padding
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        if stride is None:
            stride = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.pool = build_strided_pool(pool_type, kernel_size, stride=stride, padding=padding, dilation=dilation)

    def calc_output_size(self, input_size):
        """
        NOTE: This is designed for pytorch longtensors, if you pass an integer, make sure to cast it back to an
        integer as python3 will perform float division on it
        """
        output_size = (input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
        return output_size

    def calc_input_size(self, output_size):
        return (output_size - 1) * self.stride - 2 * self.padding + 1 + self.dilation * (self.kernel_size - 1)

    def forward(self, x):
        # Expect x as BatchSize x Length1 x ... x LengthN x Filters
        if channels_mode == "channels_last":
            return self.unfix_input(self.pool(self.fix_input(x)))
        else:
            return self.pool(x)

    def fix_input(self, x):
        raise NotImplementedError()

    def unfix_input(self, x):
        raise NotImplementedError()

    def __str__(self):
        return "%r" % self.pool


class Strided1D(StridedPool):

    def fix_input(self, inputs):
        return inputs.transpose(1, 2)

    def unfix_input(self, outputs):
        return outputs.transpose(1, 2)


class Strided2D(StridedPool):

    def fix_input(self, inputs):
        return inputs.permute(0, 3, 1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.permute(0, 2, 3, 1).contiguous()


class MaxPooling1D(Strided1D):

    def __init__(self, kernel_size, stride=None, padding='same', dilation=1):
        super(MaxPooling1D, self).__init__("max1d", kernel_size, stride=stride, padding=padding, dilation=dilation)


class SequenceMaxPooling1D(MaxPooling1D):

    def forward(self, seq_inputs):
        return [super(SequenceMaxPooling1D, self).forward(sample.unsqueeze(0)).squeeze(0) for sample in seq_inputs]


class AveragePooling1D(Strided1D):

    def __init__(self, kernel_size, stride=None, padding='same', dilation=1):
        super(AveragePooling1D, self).__init__("avg1d", kernel_size, stride=stride, padding=padding, dilation=dilation)


class MaxPooling2D(Strided2D):
    def __init__(self, kernel_size, stride=None, padding='same', dilation=1):
        super(MaxPooling2D, self).__init__("max2d", kernel_size, stride=stride, padding=padding, dilation=dilation)


class GlobalMaxPooling1D(Layer):

    def __init__(self):
        super(GlobalMaxPooling1D, self).__init__()

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def calc_output_size(self, input_size):
        return input_size / input_size

    def forward(self, x):
        # The input comes in as B x L x E
        return torch.max(x, dim=1)[0]


class SequenceGlobalMaxPooling1D(Layer):

    def __init__(self):
        super(SequenceGlobalMaxPooling1D, self).__init__()

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def calc_output_size(self, input_size):
        return input_size / input_size

    def forward(self, x):
        # The input comes in as B x Li x E
        return torch.stack([torch.max(seq, dim=0)[0] for seq in x])


class GlobalAveragePooling1D(Layer):

    def __init__(self):
        super(GlobalAveragePooling1D, self).__init__()

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def calc_output_size(self, input_size):
        return input_size / input_size

    def forward(self, x):
        # The input comes in as B x L x E
        return torch.mean(x, dim=1)


class SequenceGlobalAveragePooling1D(Layer):

    def __init__(self):
        super(SequenceGlobalAveragePooling1D, self).__init__()

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def calc_output_size(self, input_size):
        return input_size / input_size

    def forward(self, x):
        # The input comes in as B x Li x E
        return torch.stack([torch.mean(seq, dim=0) for seq in x])


class KMaxPooling1D(Layer):

    def __init__(self, k):
        super(KMaxPooling1D, self).__init__()
        self.k = k

        # Logging
        logging.info("Creating layer: {}".format(str(self)))

    def calc_output_size(self, input_size):
        return self.k * input_size / input_size

    def forward(self, x):
        # B x L x E
        return kmax_pooling(x, 1, self.k)

    def __str__(self):
        return self.__class__.__name__ + "(k=%s)" % self.k


# convolutional.py
# TODO: Add padding and cropping layers


def build_conv(dimensions, input_size, output_size, kernel_size, stride=1,
               dilation=1, groups=1, use_bias=True, input_activation='linear',
               activation='linear', num_layers=1,
               input_batchnorm=False, batchnorm=False,
               input_dropout=0.0, dropout=0.0):
    # Create the sequential
    layer = nn.Sequential()
    # Add the input dropout
    if input_dropout:
        layer.add_module(
            name="input-dropout",
            module=nn.Dropout(input_dropout))
    if input_batchnorm:
        layer.add_module(
            name="input-batchnorm",
            module=Conv.bn_constructors[dimensions](input_size))
    if input_activation != 'linear':
        try:
            layer.add_module(
                name="input_{}".format(input_activation),
                module=get_activation_type(input_activation)(inplace=True)
            )
        except TypeError:  # If inplace is not an option on the activation
            layer.add_module(
                name="input_{}".format(input_activation),
                module=get_activation_type(input_activation)())
    # Add each layer
    for i in range(num_layers):
        layer_input = input_size if i == 0 else output_size
        layer.add_module(name="conv-%s" % i,
                         module=Conv.layer_constructors[dimensions](
                            layer_input, output_size, kernel_size,
                            stride=stride, dilation=dilation, groups=groups,
                            bias=use_bias))
        if activation != "linear":
            try:
                layer.add_module(
                    name="{}-{}".format(activation, i),
                    module=get_activation_type(activation)(inplace=True)
                )
            except TypeError:  # If inplace is not an option on the activation
                layer.add_module(
                    name="{}-{}".format(activation, i),
                    module=get_activation_type(activation)()
                )
        if batchnorm:
            layer.add_module(
                name="batchnorm-%s" % i,
                module=Conv.bn_constructors[dimensions](output_size))
        if dropout:
            layer.add_module(name="dropout-%s" % i, module=nn.Dropout(dropout))

    logging.info("Creating layers: %r" % layer)
    return layer


class Conv(Layer):

    layer_constructors = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    bn_constructors = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}

    def __init__(self, dimensions, filters, kernel_size, input_shape=None,
                 stride=1, padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear', num_layers=1,
                 input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv, self).__init__()
        # Catch any bad padding inputs (NOTE: this does not catch negative padding)
        if padding != 'same' and not isinstance(padding, int):
            raise NotImplementedError("padding: %s" % padding)
        if dimensions not in [1, 2, 3]:
            raise NotImplementedError("Conv{}D".format(dimensions))

        # Set up attributes
        self.dimensions = dimensions
        self.filters = filters
        self.input_shape = input_shape
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.input_activation = input_activation
        self.activation = activation
        self.num_layers = num_layers
        self.input_batchnorm = input_batchnorm
        self.batchnorm = batchnorm
        self.input_dropout = input_dropout
        self.dropout = dropout

        # Build the layers
        self.conv_layers = []

    def get_same_padding(self, input_len):
        total_padding = int(self.stride * (input_len - 1) + 1 + self.dilation * (self.kernel_size - 1) - input_len)
        if total_padding % 2 == 1:
            pad_l = total_padding // 2
            return pad_l, total_padding - pad_l
        else:
            pad = total_padding // 2
            return pad, pad

    def get_padding(self, input_len):
        if self.padding != 'same':
            return self.padding, self.padding
        else:
            return self.get_same_padding(input_len)

    def pad_input(self, x):
        raise NotImplementedError("Layer does not know how to pad input")

    @builder
    def __build_layer(self, inputs):
        if self.input_shape is None:
            self.input_shape = get_input_shape(inputs)
        if channels_mode == "channels_last":
            input_channels = self.input_shape[-1]
        else:
            input_channels = self.input_shape[0]
        self.conv_layers = build_conv(
            self.dimensions, input_channels, self.filters,
            self.kernel_size, stride=self.stride,
            dilation=self.dilation, groups=self.groups, use_bias=self.use_bias,
            input_activation=self.input_activation, activation=self.activation,
            num_layers=self.num_layers, input_batchnorm=self.input_batchnorm,
            batchnorm=self.batchnorm, input_dropout=self.input_dropout,
            dropout=self.dropout)

    def calc_output_size(self, input_size):
        """
        NOTE: This is designed for pytorch longtensors, if you pass an integer, make sure to cast it back to an
        integer as python3 will perform float division on it
        """
        output_size = (input_size - self.dilation * (self.kernel_size - 1) + 2 * self.padding - 1) // self.stride + 1
        return output_size

    def calc_input_size(self, output_size):
        return (output_size - 1) * self.stride - 2 * self.padding + 1 + self.dilation * (self.kernel_size - 1)

    def forward(self, inputs):
        if not self.built:
            self.__build_layer(inputs)
        # Expect inputs as BatchSize x Length1 x ... x LengthN x Filters
        if channels_mode == "channels_last":
            inputs = self.fix_input(inputs)
        inputs = self.conv_layers(self.pad_input(inputs))
        if channels_mode == "channels_last":
            inputs = self.unfix_input(inputs)
        return inputs

    def reset_parameters(self):
        for layer in self.conv_layers:
            if any(isinstance(layer, self.layer_constructors[dim]) or isinstance(layer, self.bn_constructors[dim])
                   for dim in self.layer_constructors):
                logging.info("Resetting layer %s" % layer)
                layer.reset_parameters()

    def __str__(self):
        return "%r" % self.conv_layers


class Conv1D(Conv):
    def __init__(self, filters, kernel_size, input_shape=None, stride=1,
                 padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear',
                 num_layers=1, input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv1D, self).__init__(1, filters, kernel_size,
                                     input_shape=input_shape, stride=stride,
                                     padding=padding,
                                     dilation=dilation, groups=groups,
                                     use_bias=use_bias,
                                     input_activation=input_activation,
                                     activation=activation,
                                     num_layers=num_layers,
                                     input_batchnorm=input_batchnorm,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout,
                                     dropout=dropout)

    def fix_input(self, inputs):
        return inputs.transpose(1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.transpose(1, 2).contiguous()

    def pad_input(self, inputs):
        # inputs is batch_size x channels x length
        return F.pad(inputs, self.get_padding(inputs.size(2)))


class Conv2D(Conv):
    def __init__(self, filters, kernel_size, input_shape=None, stride=1,
                 padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear',
                 num_layers=1, input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv2D, self).__init__(2, filters, kernel_size,
                                     input_shape=input_shape, stride=stride,
                                     padding=padding,
                                     dilation=dilation, groups=groups,
                                     use_bias=use_bias,
                                     input_activation=input_activation,
                                     activation=activation,
                                     num_layers=num_layers,
                                     input_batchnorm=input_batchnorm,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout,
                                     dropout=dropout)

    def fix_input(self, inputs):
        return inputs.permute(0, 3, 1, 2).contiguous()

    def unfix_input(self, outputs):
        return outputs.permute(0, 2, 3, 1).contiguous()

    def pad_input(self, inputs):
        # inputs is batch_size x channels x height x width
        padding = self.get_padding(inputs.size(2)) + \
                  self.get_padding(inputs.size(3))
        return F.pad(inputs, padding)


class Conv3D(Conv):
    def __init__(self, filters, kernel_size, input_shape=None, stride=1,
                 padding='same', dilation=1, groups=1,
                 use_bias=True, input_activation='linear', activation='linear',
                 num_layers=1, input_batchnorm=False, batchnorm=False,
                 input_dropout=0.0, dropout=0.0):
        super(Conv3D, self).__init__(3, filters, kernel_size,
                                     input_shape=input_shape, stride=stride,
                                     padding=padding,
                                     dilation=dilation, groups=groups,
                                     use_bias=use_bias,
                                     input_activation=input_activation,
                                     activation=activation,
                                     num_layers=num_layers,
                                     input_batchnorm=input_batchnorm,
                                     batchnorm=batchnorm,
                                     input_dropout=input_dropout,
                                     dropout=dropout)

    def fix_input(self, inputs):
        return inputs.permute(0, 4, 1, 2, 3).contiguous()

    def unfix_input(self, outputs):
        return outputs.permute(0, 2, 3, 4, 1).contiguous()

    def pad_input(self, inputs):
        # inputs is batch_size x channels x height x width x time
        padding = self.get_padding(inputs.size(2)) + \
                  self.get_padding(inputs.size(3)) + \
                  self.get_padding(inputs.size(4))
        return F.pad(inputs, padding)


# recurrent.py
def build_rnn(rnn_type, input_size, output_size, num_layers=1, bidirectional=False,
              input_dropout=0.0, dropout=0.0):
    # Create the sequential
    layer = nn.Sequential()
    # Add the input dropout
    if input_dropout:
        layer.add_module(name="input-dropout", module=nn.Dropout(input_dropout))
    layer.add_module(name="rnn", module=RNN.layer_constructors[rnn_type](input_size, output_size, num_layers=num_layers, dropout=dropout,
                                                      bidirectional=bidirectional, batch_first=True))
    logging.info("Creating layer: %r" % layer)
    return layer


class RNN(Layer):

    layer_constructors = {'gru': nn.GRU, 'lstm': nn.LSTM,
                          "tanh_simple": lambda *args, **kwargs: nn.RNN(*args, nonlinearity='tanh', **kwargs),
                          "relu_simple": lambda *args, **kwargs: nn.RNN(*args, nonlinearity='relu', **kwargs)}

    def __init__(self, rnn_type, units, input_shape=None, num_layers=1,
                 bidirectional=False, input_dropout=0.0, dropout=0.0,
                 return_sequences=False, return_state=False):
        super(RNN, self).__init__()
        units = units // 2 if bidirectional else units

        # Set up the attributes
        self.rnn_type = rnn_type
        self.input_shape = input_shape
        self.units = units
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Build the layers
        self.rnn_layers = []

    @builder
    def __build_layer(self, inputs):
        if self.input_shape is None:
            self.input_shape = get_input_shape(inputs)
        self.rnn_layers = build_rnn(
            self.rnn_type, self.input_shape[-1], self.units,
            num_layers=self.num_layers, bidirectional=self.bidirectional,
            input_dropout=self.input_dropout, dropout=self.dropout)

    def calc_output_size(self, input_size):
        return input_size

    def forward(self, x):
        if not self.built:
            self.__build_layer(x)
        x, states = self.rnn_layers(x)
        if not self.return_sequences:
            if self.bidirectional:
                x = torch.cat([x[:, -1, :self.units], x[:, 0, self.units:]], dim=-1)
            else:
                x = x[:, -1]
        if self.return_state:
            return x, states
        return x

    def reset_parameters(self):
        for layer in self.rnn_layers:
            if isinstance(layer, nn.RNNBase):
                logging.info("Resetting layer %s" % layer)
                layer.reset_parameters()

    def __str__(self):
        return ("%r\n\treturn_sequences={}, return_state={}" % self.rnn_layers).format(self.return_sequences,
                                                                                       self.return_state)


class SimpleRNN(RNN):
    def __init__(self, units, input_shape=None, num_layers=1,
                 bidirectional=False, input_dropout=0.0, dropout=0.0,
                 return_sequences=False, return_state=False,
                 nonlinearity='tanh'):
        rnn_type = nonlinearity + "_" + "simple"
        super(SimpleRNN, self).__init__(
            rnn_type, units, input_shape=input_shape, num_layers=num_layers,
            bidirectional=bidirectional, input_dropout=input_dropout,
            dropout=dropout, return_sequences=return_sequences,
            return_state=return_state)


class GRU(RNN):
    def __init__(self, units, input_shape=None, num_layers=1,
                 bidirectional=False, input_dropout=0.0, dropout=0.0,
                 return_sequences=False, return_state=False):
        super(GRU, self).__init__(
            'gru', units, input_shape=input_shape, num_layers=num_layers,
            bidirectional=bidirectional, input_dropout=input_dropout,
            dropout=dropout, return_sequences=return_sequences,
            return_state=return_state)


class LSTM(RNN):
    def __init__(self, units, input_shape=None, num_layers=1,
                 bidirectional=False, input_dropout=0.0, dropout=0.0,
                 return_sequences=False, return_state=False):
        super(LSTM, self).__init__(
            'lstm', units, input_shape=input_shape, num_layers=num_layers,
            bidirectional=bidirectional, input_dropout=input_dropout,
            dropout=dropout, return_sequences=return_sequences,
            return_state=return_state)


# wrappers.py
class Identity(Layer):
    """
    This is used to create layer wrappers without passing a layer.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

    def calc_output_size(self, input_size):
        return input_size


# Singleton Identity layer
Identity = Identity()


class SequenceInput(Layer):
    """
    Wrapper for a layer that should take in variable length sequences as inputs.
    This wrapper will take as input a list of (batch size number of) sequences.
    Before passing to its layer, the wrapper will pad the sequences to the longest
    sequence in the batch, pass to the layer, then unpad back to the list of sequence form.
    The wrapper requires that sequence lengths are not modified when passed through the layer.

    Dropout will be applied to the nonpadded sequence.
    """
    def __init__(self, wrapped_layer=Identity, input_dropout=0., dropout=0., pad_value=0.):
        super(SequenceInput, self).__init__()
        self.layer = wrapped_layer
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_dropout_p = input_dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.pad_value = pad_value
        self.__descriptor = "SequenceInput(input_dropout=%s, dropout=%s, pad_value=%s)" % (
                             self.input_dropout, self.dropout, self.pad_value)
        logging.info("Wrapping layer with %s: %r" % (self.__descriptor, self.layer))

    def forward(self, x):
        if self.input_dropout_p:
            x = [self.input_dropout(sample.unsqueeze(0)).squeeze(0) for sample in x]
        x, pad_lens = pad_sequences(x, pad_value=self.pad_value)
        x = self.layer(x)
        x = unpad_sequences(x, pad_lens)
        if self.dropout_p:
            x = [self.dropout(sample.unsqueeze(0)).squeeze(0) for sample in x]
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()

    def __str__(self):
        return self.__descriptor + "(%r)" % self.layer


class TimeDistributed(Layer):

    def __init__(self, wrapped_layer):
        super(TimeDistributed, self).__init__()
        self.layer = wrapped_layer
        logging.info("TimeDistributing %r layer" % self.layer)

    def forward(self, x):
        x, seq_lens = pack_sequences(x)  # B*Li x I
        x = self.layer(x)  # B*Li x O
        x = unpack_sequences(x, seq_lens)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()

    def __str__(self):
        return "TimeDistributed" + "(%r)" % self.layer


class MaskedLayer(Layer):

    def __init__(self, layer=Identity, mask_value=0.0, dim=1):
        super(MaskedLayer, self).__init__()
        self.layer = layer
        self.dim = dim
        if dim == 1:
            self.masker = MaskedInput(mask_value=mask_value)
        elif dim == 2:
            self.masker = MaskedInput2D(mask_value=mask_value)
        else:
            raise NotImplementedError("dim=%s" % dim)
        logging.info("Masking {} layer with mask_value={}".format(self.layer, mask_value))

    def forward(self, x, seq_lens):
        x = self.masker(x, seq_lens)
        x = self.layer(x)
        seq_lens = self.layer.calc_output_size(seq_lens)
        return x, seq_lens

    def reset_parameters(self):
        self.layer.reset_parameters()


# attentional.py
class ContextAttention(Layer):

    def __init__(self, units, input_shape=None, activation='tanh',
                 batchnorm=False, padded_input=True, dropout=0.0):
        super(ContextAttention, self).__init__()
        self.units = units
        self.input_shape = input_shape
        self.activation_name = activation
        self.batchnorm = batchnorm
        self.padded_input = padded_input
        self.dropout = dropout

        self.attentional_module = None
        self.context_vector = None
        self.context_attention = None

    @builder
    def __build_layer(self, inputs):
        assert self.input_shape is None
        # Use the 0th input since the inputs are time distributed
        self.input_shape = get_input_shape(inputs[0])
        self.attentional_module = FullyConnected(
            self.input_shape[0], input_shape=self.input_shape,
            activation=self.activation_name, batchnorm=self.batchnorm,
            dropout=self.dropout)
        self.context_vector = FullyConnected(
            self.units, input_shape=self.input_shape, use_bias=False,
            batchnorm=False)
        self.context_attention = TimeDistributed(
            nn.Sequential(self.attentional_module, self.context_vector)
        )

    def forward(self, x, seq_lens=None):
        if self.padded_input:
            padded_input = x
            if seq_lens is None:
                seq_lens = LongTensor([x.size(1)] * x.size(0))
            x = unpad_sequences(x, seq_lens)
        else:
            padded_input, _ = pad_sequences(x)  # B x L x H
        # Build the layer if we don't know the input shape
        if not self.built:
            self.__build_layer(x)

        # The input comes in as B x Li x E
        att = self.context_attention(x)  # B x L x H
        att, _ = seq_softmax(att, return_padded=True)  # B x L x K
        out = torch.bmm(att.transpose(1, 2), padded_input)  # B x K x H
        return out.squeeze_(1)

    def reset_parameters(self):
        if self.built:
            self.attentional_module.reset_parameters()
            self.context_vector.reset_parameters()

    def __str__(self):
        return "%r" % self.pool


class ContextMaxPool1D(Layer):

    def __init__(self, units=1, input_shape=None, activation='linear',
                 batchnorm=False, padded_input=True, dropout=0.0):
        super(ContextMaxPool1D, self).__init__()
        self.units = units
        self.activation = activation
        self.batchnorm = batchnorm
        self.padded_input = padded_input
        self.dropout = dropout

        self.max_pool = SequenceGlobalMaxPooling1D()
        self.context_attention = None


    @builder
    def __build_layer(self, inputs):
        assert self.input_shape is None
        # Use the 0th input since the inputs are time distributed
        self.input_shape = get_input_shape(inputs[0])
        self.context_attention = nn.ModuleList(
            [TimeDistributed(
                FullyConnected(self.input_shape[0],
                    input_shape=self.input_shape, batchnorm=self.batchnorm,
                    activation=self.activation, dropout=self.dropout
                ) for _ in range(self.units))
            ]
        )

    def forward(self, x, seq_lens=None):
        if self.padded_input:
            padded_input = x
            x = unpad_sequences(x, seq_lens)
        else:
            padded_input, _ = pad_sequences(x)  # B x L x H

        # Build the layer if we don't know the input shape
        if not self.built:
            self.__build_layer(x)
        # The input comes in as B x Li x E
        out_heads = torch.stack([self.max_pool(head(x)) for head in self.context_attention], dim=1)  # B x K x H
        return out_heads.squeeze_(1)

    def reset_parameters(self):
        if self.built:
            for i in range(len(self.context_attention)):
                self.context_attention[i].reset_parameters()

    def __str__(self):
        return "%r" % self.pool


"""
AUGMENTERS
Files included:
    - augmenters.py
Files not included:
    image.py
"""


class Augmenter(object):

    def __init__(self, labels=True, augment_labels=False):
        self.labels = labels
        self.augment_labels = augment_labels

    def _augment(self, batch):
        # Split the batch if necessary
        if self.labels:
            x, y = batch
            seed = np.random.randint(2 ** 32)
            if self.augment_labels:
                np.random.seed(seed)
                y = self.augment(y)
            np.random.seed(seed)
            x = self.augment(x)
            return x, y

        else:
            x = batch
            return self.augment(x)

    def augment(self, x):
        raise NotImplementedError()

    def __call__(self, generator):
        return AugmenterGenerator(self, generator)


class AugmenterGenerator(BatchGenerator):

    def __init__(self, augmenter, generator):
        # Copy the steps per epoch and batch size if it has one
        if hasattr(generator, "steps_per_epoch") \
                and hasattr(generator, "batch_size"):
            super(AugmenterGenerator, self).__init__(
                steps_per_epoch=generator.steps_per_epoch,
                batch_size=generator.batch_size)
        else:
            logging.warning("Input generator does not have a "
                            "steps_per_epoch or batch_size "
                            "attribute. Continuing without them.")

        self.augmenter = augmenter
        self.generator = generator

    def __next__(self):
        return self.augmenter._augment(next(self.generator))
