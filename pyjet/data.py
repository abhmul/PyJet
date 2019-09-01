import logging
import threading
import numpy as np

# For image dataset
from skimage.transform import resize
from skimage.io import imread
from skimage.color import rgb2ycbcr, rgb2gray, gray2rgb

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
        return float("inf")

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

    def flow(
        self,
        steps_per_epoch=None,
        batch_size=None,
        shuffle=True,
        replace=False,
        seed=None,
    ):
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
            seed=seed,
        )

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

    def __init__(
        self,
        dataset,
        steps_per_epoch=None,
        batch_size=None,
        shuffle=True,
        replace=False,
        seed=None,
    ):
        super(DatasetGenerator, self).__init__(steps_per_epoch, batch_size)
        self.dataset = dataset
        self.shuffle = shuffle
        self.replace = replace
        self.seed = seed
        self.index_array = None
        self.lock = threading.Lock()

        # Some input checking
        check = (
            int(self.steps_per_epoch is not None)
            + int(self.batch_size is not None)
            + int(len(self.dataset) != float("inf"))
        )
        if check < 2:
            raise ValueError(
                "2 of the following must be defined: len(dataset),"
                " steps_per_epoch, and batch_size."
            )
        # Otherwise, we're good, infer the missing info
        if len(self.dataset) != float("inf"):
            self.index_array = np.arange(len(self.dataset))
        if self.batch_size is None:
            if self.steps_per_epoch is None:
                raise ValueError()
            self.batch_size = int(
                (len(self.dataset) + self.steps_per_epoch - 1) / self.steps_per_epoch
            )
        if self.steps_per_epoch is None:
            self.steps_per_epoch = int(
                (len(self.dataset) + self.batch_size - 1) / self.batch_size
            )
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
                    yield (np.random.choice(self.index_array, self.batch_size, True),)
                else:
                    yield (self.index_array[i : i + self.batch_size],)

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
        ids -- The ids corresponding to each example (optional)
        weights -- The weight to apply to each label
    """

    def __init__(self, x, y=None, ids=None, weights=None):
        super(NpDataset, self).__init__()

        self.x = x
        self.y = y
        self.ids = ids
        self.weights = weights

        assert isinstance(self.x, np.ndarray), "x must be a numpy array."
        if self.y is not None:
            assert isinstance(self.y, np.ndarray), "y must be a numpy array or None."
        if self.ids is not None:
            assert isinstance(
                self.ids, np.ndarray
            ), "ids must be a numpy array or None."
        if self.weights is not None:
            assert isinstance(
                self.weights, np.ndarray
            ), "weights must be a numpy or None."

        self.output_labels = self.has_labels
        if self.has_labels:
            assert len(self.x) == len(self.y), (
                "Data and labels must have same number of" + "samples. X has shape ",
                len(x),
                " and Y has shape ",
                len(y),
                ".",
            )
        if self.has_ids:
            assert len(self.x) == len(self.ids), (
                "Data and ids must have same number of" + "samples. X has shape ",
                len(x),
                " and ids has shape ",
                len(ids),
                ".",
            )

        self.output_weights = self.output_labels and self.has_weights
        if self.has_weights:
            assert len(self.x) == len(self.weights), (
                "Data and weights must have same number of samples. X has shape ",
                len(x),
                " and weights has shape ",
                len(weights),
                ".",
            )

    def __len__(self):
        return len(self.x)

    @property
    def has_ids(self):
        return self.ids is not None

    @property
    def has_weights(self):
        return self.weights is not None

    @property
    def has_labels(self):
        return self.y is not None

    def toggle_labels(self):
        self.output_labels = not self.output_labels

    def create_batch(self, batch_indicies):
        batch_x = self.x[batch_indicies]
        if self.output_labels:
            batch_y = self.y[batch_indicies]
            if self.output_weights:
                batch_weights = self.weights[batch_indicies]
                return batch_x, (batch_y, batch_weights)
            return batch_x, batch_y
        return batch_x

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
                label_mask = np.all(stratify_by == unq_label, axis=non_batch_dims)
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

    def get_split_indicies(self, split, shuffle, seed, stratified, stratify_by):
        if stratified:
            if stratify_by is None:
                stratify_by = self.y
            assert stratify_by is not None, "Data must have labels to " "stratify by."
            assert len(stratify_by) == len(self), (
                "Labels to stratify by " "have same length as the dataset."
            )

        if shuffle:
            if seed is not None:
                np.random.seed(seed)

        if stratified:
            train_split, val_split = self.get_stratified_split_indicies(
                split, shuffle, seed, stratify_by
            )
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

    def validation_split(
        self, split=0.2, shuffle=False, seed=None, stratified=False, stratify_by=None
    ):
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
            split, shuffle, seed, stratified, stratify_by
        )
        train_data = self.__class__(
            self.x[train_split],
            y=None if not self.has_labels else self.y[train_split],
            ids=None if not self.has_ids else self.ids[train_split],
        )
        val_data = self.__class__(
            self.x[val_split],
            y=None if not self.has_labels else self.y[val_split],
            ids=None if not self.has_ids else self.ids[val_split],
        )
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
            train_data = self.__class__(
                self.x[train_split],
                y=None if not self.has_labels else self.y[train_split],
                ids=None if not self.has_ids else self.ids[train_split],
            )
            val_data = self.__class__(
                self.x[val_split],
                y=None if not self.has_labels else self.y[val_split],
                ids=None if not self.has_ids else self.ids[val_split],
            )
            yield train_data, val_data


# TODO: Add weights to other NPDataset models
class ImageDataset(NpDataset):
    """
        A Dataset that is built from a numpy array of filenames for images

        # Arguments
            x -- The input data as a numpy array of filenames
            y -- The target data as a numpy array of labels (optional)
        """

    MODE2FUNC = {
        "rgb": lambda x: x,
        "ycbcr": rgb2ycbcr,
        "gray": lambda x: rgb2gray(x)[:, :, np.newaxis],
    }

    def __init__(self, x, y=None, ids=None, img_size=None, mode="rgb"):
        super(ImageDataset, self).__init__(x, y=y, ids=ids)
        self.img_size = img_size
        assert mode in ImageDataset.MODE2FUNC, "Invalid mode %s" % mode
        self.mode = mode
        logging.info(
            "Creating ImageDataset(img_size={img_size}, mode={mode}".format(
                img_size=self.img_size, mode=self.mode
            )
        )

    @staticmethod
    def load_img(path_to_img, img_size=None, mode="rgb", to_float=True):
        img = imread(path_to_img)
        # Then its a grayscale image
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        elif np.all(img[:, :, 0:1] == img):
            img = img[:, :, 0:1]
        # Otherwise it's rgb
        else:
            # Cut out the alpha channel
            img = img[:, :, :3]
            img = ImageDataset.MODE2FUNC[mode](img)

        orig_img_shape = img.shape[:2]
        # Resize to the input size
        if img_size is not None and img_size != orig_img_shape:
            img = resize(img, img_size, mode="constant", preserve_range=True).astype(
                np.uint8
            )
        if to_float:
            # Normalize the image
            if mode == "rgb":
                img = img / 255.0
            elif mode == "ycbcr":
                img = img / 235.0
            elif mode == "gray":
                img = img / 255.0
        else:
            # Use uint8 to keep mem low
            img = img.astype(np.uint8, copy=False)

        return img, orig_img_shape

    @staticmethod
    def _load_img_batch_known_size(
        img_paths, img_size, mode="rgb", to_float=True, progress=False
    ):
        n = len(img_paths)
        dtype = np.float32 if to_float else np.uint8

        # Load the first image to get the specs on it
        img0, img0_shape = ImageDataset.load_img(
            img_paths[0], img_size=img_size, mode=mode, to_float=to_float
        )

        images = np.empty((n, *(img0.shape)), dtype=dtype)
        img_shapes = np.empty((n, 2), dtype=int)
        img_shapes[0] = img0_shape
        images[0] = img0
        # Load the rest of the images

        for i in range(1, n):
            images[i], img_shapes[i] = ImageDataset.load_img(
                img_paths[i], img_size=img_size, mode=mode, to_float=to_float
            )

        return images, img_shapes

    @staticmethod
    def _load_img_batch_unknown_size(img_paths, mode="rgb", to_float=True):
        images, img_shapes = zip(
            *(
                ImageDataset.load_img(
                    path_to_img, img_size=None, mode=mode, to_float=to_float
                )
                for path_to_img in img_paths
            )
        )

        # If image sizes are the same, create one large np array, otherwise
        # create an array of numpy array objects
        if all(img_shapes[0] == img_shape for img_shape in img_shapes):
            images = np.stack(images)
        else:
            images = np.array(images, dtype="O")

        # Turn the img shapes into an array
        img_shapes = np.array(img_shapes)

        return images, img_shapes

    @staticmethod
    def load_img_batch(img_paths, img_size=None, mode="rgb", to_float=True):
        if img_size is None:
            return ImageDataset._load_img_batch_unknown_size(
                img_paths, mode=mode, to_float=to_float
            )
        else:
            return ImageDataset._load_img_batch_known_size(
                img_paths, img_size=img_size, mode=mode, to_float=to_float
            )

    def create_batch(self, batch_indicies):
        filenames = self.x[batch_indicies]
        x = self.load_img_batch(filenames, img_size=self.img_size, mode=self.mode)[0]

        if self.output_labels:
            return x, self.y[batch_indicies]
        return x


class ImageMaskDataset(ImageDataset):
    """
        A Dataset that is built from a numpy array of filenames for images

        # Arguments
            x -- The input data as a numpy array of filenames
            y -- The target data as a numpy array of filenames (optional)
    """

    def create_batch(self, batch_indicies):
        filenames = self.x[batch_indicies]
        x = self.load_img_batch(filenames, img_size=self.img_size, mode=self.mode)[0]

        if self.output_labels:
            # Load the output masks
            filenames = self.y[batch_indicies]
            y = self.load_img_batch(filenames, img_size=self.img_size, mode="gray")[0][
                :, :, :, 0
            ]
            return x, y

        return x


class HDF5Dataset(Dataset):
    def __init__(self, x, y=None):
        super(HDF5Dataset, self).__init__()
        raise NotImplementedError()


class TorchDataset(Dataset):
    def __init__(self, x, y=None):
        super(TorchDataset, self).__init__()
        raise NotImplementedError()
