import logging

from . import data


class Augmenter(object):

    def augment(self, batch):
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
        return self.augmenter.augment(next(self.generator))
