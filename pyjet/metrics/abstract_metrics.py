class Metric(object):
    """
    The abstract metric that defines the metric API. Some notes on it:

    - Passing a function of the form `metric(y_true, y_pred)` to an abstract
    metric will use that function to calculate the score on a batch.

    - The accumulate method is called at the end of each batch to `accumulate`
    the scores over all the previous batches. It receives the calculated
    score for the batch, but does not need to use itself.
        - See `AverageMetric` for an example that uses the input value to
        accumulate.
        - See `Accuracy` for an example that does not use the input value
        to accumulate.

    - The reset method is called at the end of each epoch or validation run. It
    simply creates a new version of the metric to reset its internal state and
    returns this reset metric. Feel free to override it with your own reset
    behavior as long as it returns a "reset" version of your metric.
    """
    def __init__(self, metric_func=None):
        self.metric_func = metric_func

    def __call__(self, y_true, y_pred):
        if self.metric_func is not None:
            return self.metric_func(y_true, y_pred)
        else:
            raise NotImplementedError()

    def accumulate(self, value=None):
        raise NotImplementedError()

    def reset(self):
        return self.__class__(metric_func=self.metric_func)


class AverageMetric(Metric):
    """
    An abstract metric that accumulates the batch values from the metric
    by averaging them together. If any function is input into the fit
    function as a metric, it will automatically be considered an AverageMetric.
    """
    def __init__(self, metric_func=None):
        super(AverageMetric, self).__init__(metric_func=metric_func)
        self.metric_sum = 0.
        self.count = 0

    def accumulate(self, value):
        self.metric_sum += value
        self.metric_count += 1
        return self.metric_sum / self.metric_count
