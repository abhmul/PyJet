import logging
import numpy as np

from . import data


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


class AugmenterGenerator(data.BatchGenerator):

    def __init__(self, augmenter, generator):
        # Copy the steps per epoch and batch size if it has one
        if hasattr(generator, "steps_per_epoch") and hasattr(generator, "batch_size"):
            super(AugmenterGenerator, self).__init__(
                steps_per_epoch=generator.steps_per_epoch, batch_size=generator.batch_size)
        else:
            logging.warning("Input generator does not have a steps_per_epoch or batch_size "
                            "attribute. Continuing without them.")

        self.augmenter = augmenter
        self.generator = generator

    def __next__(self):
        return self.augmenter._augment(next(self.generator))
