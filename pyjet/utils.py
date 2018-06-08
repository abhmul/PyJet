import copy


def resettable(f):
    """
    Decorator to make a python object resettable. Note that this will
    no work with inheritance. To reset an object, simply call its reset
    method.
    """
    def __init_and_copy__(self, *args, **kwargs):
        f(self, *args)
        self.__original_dict__ = copy.deepcopy(self.__dict__)

        def reset(o=self):
            o.__dict__ = o.__original_dict__

        self.reset = reset

    return __init_and_copy__
