import copy
import os
import logging


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
