METRICS_REGISTRY = {}


def register_metric(name, metric):
    metric.__name__ = name
    METRICS_REGISTRY[name] = metric


def load_metric(name):
    return METRICS_REGISTRY[name]
