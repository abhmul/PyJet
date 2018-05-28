import time
from collections import deque
import numpy as np
import warnings
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None


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

    def on_epoch_begin(self, epoch, train_logs=None, val_logs=None):
        """Called at the start of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(
                epoch, train_logs=train_logs, val_logs=val_logs)
        self._delta_t_batch = 0.
        self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
        self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

    def on_epoch_end(self, epoch, train_logs=None, val_logs=None):
        """Called at the end of an epoch.
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(
                epoch, train_logs=train_logs, val_logs=val_logs)

    def on_batch_begin(self, batch, train_logs=None, val_logs=None):
        """Called right before processing a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_begin(
                batch, train_logs=train_logs, val_logs=val_logs)
        self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_begin)
        if (self._delta_t_batch > 0.
                and delta_t_median > 0.95 * self._delta_t_batch
                and delta_t_median > 0.1):
            warnings.warn('Method on_batch_begin() is slow compared '
                          'to the batch update (%f). Check your callbacks.' %
                          delta_t_median)
        self._t_enter_batch = time.time()

    def on_batch_end(self, batch, train_logs=None, val_logs=None):
        """Called at the end of a batch.
        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        if not hasattr(self, '_t_enter_batch'):
            self._t_enter_batch = time.time()
        self._delta_t_batch = time.time() - self._t_enter_batch
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            callback.on_batch_end(
                batch, train_logs=train_logs, val_logs=val_logs)
        self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
        delta_t_median = np.median(self._delta_ts_batch_end)
        if (self._delta_t_batch > 0.
                and (delta_t_median > 0.95 * self._delta_t_batch
                     and delta_t_median > 0.1)):
            warnings.warn('Method on_batch_end() is slow compared '
                          'to the batch update (%f). Check your callbacks.' %
                          delta_t_median)

    def on_train_begin(self, train_logs=None, val_logs=None):
        """Called at the beginning of training.
        # Arguments
            logs: dictionary of logs.
        """
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(train_logs=train_logs, val_logs=val_logs)

    def on_train_end(self, train_logs=None, val_logs=None):
        """Called at the end of training.
        # Arguments
            logs: dictionary of logs.
        """
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        for callback in self.callbacks:
            callback.on_train_end(train_logs=train_logs, val_logs=val_logs)

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

    def on_epoch_begin(self, epoch, train_logs=None, val_logs=None):
        pass

    def on_epoch_end(self, epoch, train_logs=None, val_logs=None):
        pass

    def on_batch_begin(self, batch, train_logs=None, val_logs=None):
        pass

    def on_batch_end(self, batch, train_logs=None, val_logs=None):
        pass

    def on_train_begin(self, train_logs=None, val_logs=None):
        pass

    def on_train_end(self, train_logs=None, val_logs=None):
        pass


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

    def __init__(self,
                 filepath,
                 monitor,
                 monitor_val=True,
                 verbose=0,
                 save_best_only=False,
                 mode='auto',
                 period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.monitor_val = monitor_val
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or 'auc' in self.monitor or \
                    self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, train_logs=None, val_logs=None):
        logs = val_logs if self.monitor_val else train_logs
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch)
            if self.save_best_only:
                current = logs[self.monitor][-1]
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                'Epoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s' % (epoch, self.monitor,
                                                         self.best, current,
                                                         filepath))
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
    def __init__(self,
                 monitor,
                 scale='linear',
                 plot_during_train=True,
                 save_to_file=None,
                 block_on_end=True):
        super().__init__()
        if plt is None:
            raise ValueError(
                "Must be able to import Matplotlib to use the Plotter.")
        self.scale = scale
        self.monitor = monitor
        self.plot_during_train = plot_during_train
        self.save_to_file = save_to_file
        self.block_on_end = block_on_end
        self.fig = plt.figure()
        self.title = "{} per Epoch".format(self.monitor)
        self.xlabel = "Epoch"
        self.ylabel = self.monitor
        self.ax = self.fig.add_subplot(
            111, title=self.title, xlabel=self.xlabel, ylabel=self.ylabel)
        self.ax.set_yscale(self.scale)
        self.x = []
        self.y_train = []
        self.y_val = []
        self.ion = self.plot_during_train

    def on_train_end(self, train_logs=None, val_logs=None):
        if self.plot_during_train:
            plt.ioff()
        if self.block_on_end:
            plt.show()
        return

    def on_epoch_end(self, epoch, train_logs=None, val_logs=None):
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        self.x.append(len(self.x))
        self.y_train.append(train_logs[self.monitor][-1])
        self.y_val.append(val_logs[self.monitor][-1])
        self.ax.clear()
        # # Set up the plot
        self.fig.suptitle(self.title)

        self.ax.set_yscale(self.scale)
        # Actually plot
        self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')

        if self.ion:
            plt.ion()
            self.ion = False

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

    def __init__(self,
                 optimizer,
                 monitor,
                 monitor_val=True,
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 epsilon=1e-4,
                 cooldown=0,
                 min_lr=0):
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
        if (self.mode == 'min'
                or (self.mode == 'auto' and 'acc' not in self.monitor)):
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
            warnings.warn('Reduce LR on plateau conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s' %
                          (self.monitor,
                           ','.join(list(logs.keys()))), RuntimeWarning)

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
                            param_group['lr'] = max(old_lr * self.factor,
                                                    self.min_lr)
                            reduced_lr = True
                    if reduced_lr:
                        self.cooldown_counter = self.cooldown
                        self.wait = 0
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: ReduceLROnPlateau reducing '
                                'learning rate by %s factor.'
                                % (epoch + 1, self.factor))

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
            print('\nEpoch %05d: LRScheduler setting lr to %s.' % (epoch + 1,
                                                                   new_lr))
