import copy
import os
import logging
import warnings

python_iterables = {list, set, tuple, frozenset}


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


def standardize_list_input(inputs):
    if type(inputs) in python_iterables:
        return list(inputs)
    return [inputs]


def destandardize_list_input(inputs):
    if type(inputs) not in python_iterables:
        return inputs
    inputs = list(inputs)
    assert len(inputs) == 1
    return inputs[0]


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def newFunc(*args, **kwargs):
        warnings.warn(
            "Call to deprecated function %s." % func.__name__,
            category=DeprecationWarning,
        )
        return func(*args, **kwargs)

    newFunc.__name__ = func.__name__
    newFunc.__doc__ = func.__doc__
    newFunc.__dict__.update(func.__dict__)
    return newFunc
