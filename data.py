import threading
import numpy as np

# TODO Create a dataset for HDF5 and Torch Tensor
# TODO Create a DatasetGenerator constructor for python generators

class Dataset(object):
    """
    An abstract container for data designed to be passed to a model.
    This container should implement create_batch. It is only necessary
    to impolement validation_split() if you use this module to split your
    data into a train and test set. Same goes for kfold()

    # Note:
        Though not forced, a Dataset is really a constant object. Once created,
        it should not be mutated in any way.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        """The length is used downstream by the generator if it is not inf."""
        return float('inf')

    def create_batch(self, *args, **kwargs):
        """
        This method creates a batch of data to be sent to a model.

        Returns:
            A batch in the form of any type that can be cast to torch tensor
            by a model (numpy, HDF5, torch tensor, etc.).
        """
        raise NotImplementedError()

    def validation_split(self, split=0.2, **kwargs):
        raise NotImplementedError()

    def kfold(self, k, **kwargs):
        raise NotImplementedError()

class DatasetGenerator(object):
    """
    An iterator to create batches for a model.

    # Arguments
        dataset -- the dataset to generate from
        steps_per_epoch -- The number of iterations in one epoch
        shuffle -- Whether or not to shuffle the dataset before each epoch
                   default: True
    """
    # TODO add threading support
    def __init__(self, dataset, steps_per_epoch=None, batch_size=None, shuffle=False, seed=None):
        self.dataset = dataset
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index_array = None
        self.lock = threading.Lock()

        # Some input checking
        check = (int(self.steps_per_epoch is not None) + \
                 int(self.batch_size is not None) + \
                 int(len(self.dataset != float("inf"))))
        if check < 2:
            raise ValueError("2 of the following must be defined: len(dataset),"
                             " steps_per_epoch, and batch_size.")
        # Otherwise, we're good, infer the missing info
        if len(self.dataset) != float('inf'):
            self.index_array = np.arange(len(self.dataset))
        if self.batch_size is None:
            if self.steps_per_epoch is None:
                raise ValueError()
            self.batch_size = (len(self.dataset) + self.steps_per_epoch - 1) / \
                              self.steps_per_epoch
        if self.steps_per_epoch = None:
            self.steps_per_epoch = (len(self.dataset) + self.batch_size - 1) / \
                                   self.batch_size

    def batch_argument_generator(self):
        # TODO I hate the structure of this. If I keep this, then it should
        # really be avoided and gens should be constructed from python
        # iterators.
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
        # Set the seed if we have one
        if self.seed is not None:
            np.random.seed(self.seed)
        while True:
            # Shuffle if we need to
            if self.shuffle:
                np.random.shuffle(self.index_array)
            for i in range(0, len(self.index_array), batch_size):
                yield (self.index_array[i:i+batch_size],)

    def __iter__(self):
        return self

    def __next__(self):
        # This is a critical section, so we lock when we need the next indicies
        with self.lock:
            batch_arguments = next(self.batch_argument_generator())
        return self.dataset.create_batch(*batch_arguments)

    def toggle_shuffle(self):
        self.shuffle = not self.shuffle

class DatasetPyGenerator(object):

    def __init__(self, pygen, steps_per_epoch):
        self.pygen = pygen
        self.steps_per_epoch = steps_per_epoch
        self.index_array = None

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
    def __init__(self, x, y=None):
        super(NpDataset, self).__init__()

        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def create_batch(self, batch_indicies):
        return self.x[batch_indicies], self.y[batch_indicies]

    def validation_split(self, split=0.2, shuffle=False, seed=None,
                         destroy_self=False):
        """
        Splits the NpDataset into two smaller datasets based on the split

        # Arguments:
            split -- The fraction of the dataset to make validation.
                     Default: 0.2
            shuffle -- Whether or not to randomly sample the validation set
                       and train set from the parent dataset. Default: False
            seed -- A seed for the random number generator (optional).
            destroy_self -- Whether or not to clean the parent dataset up from
                            memory after the train and validation sets have
                            been created. Default: False

        # Returns
            A train dataset with (1-split) fraction of the data and a validation
            dataset with split fraction of the data

        # Note
            Shuffling the dataset will at one point cause double the size of
            the dataset to be loaded into RAM. If this is an issue, I suggest
            you store your dataset on disk split up into validation and train
            so you don't do this splitting in memory. You can set the
            destroy_self flag to True if you can afford the split, but want to
            reclaim the memory from the parent dataset.
        """
        split_ind = int(split * len(self))
        train_split = slice(split_ind)
        val_split = slice(split_ind, None)
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            indicies = np.random.permutation(len(self))
            train_split = indicies[train_split]
            val_split = indicies[val_split]
        train_data = NpDataset(self.x[train_split],
                               None if self.y is None else self.y[train_split])
        val_data = NpDataset(self.x[val_split],
                             None if self.y is None else self.y[val_split])
        # Clear the memory of the parent dataset. Set to None and let GC
        # handle clean up.
        if destroy_self:
            self = None
        return train_data, val_data

class HDF5Dataset(Dataset):

    def __init__(self, x, y=None):
        super(HDF5Dataset, self).__init__()
        raise NotImplementedError()

class TorchDataset(Dataset):

    def __init__(self, x, y=None):
        super(TorchDataset, self).__init__()
        raise NotImplementedError()
