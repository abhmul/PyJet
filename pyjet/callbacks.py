import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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

    def __init__(self, filepath, monitor, monitor_val=True, verbose=0,
                 save_best_only=False,
                 mode='auto', period=1):
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
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
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
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
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

    def __init__(self, monitor, scale='linear', plot_during_train=True, save_to_file=None):
        super().__init__()
        if plt is None:
            raise ValueError("Must be able to import Matplotlib to use the Plotter.")
        self.scale = scale
        self.monitor = monitor
        self.plot_during_train = plot_during_train
        self.save_to_file = save_to_file
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_yscale(self.scale)
        self.x = []
        self.y_train = []
        self.y_val = []
        # self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')

    def on_train_end(self, train_logs=None, val_logs=None):
        plt.ioff()
        plt.show()
        return

    def on_epoch_end(self, epoch, train_logs=None, val_logs=None):
        train_logs = train_logs or {}
        val_logs = val_logs or {}
        self.x.append(len(self.x))
        self.y_train.append(train_logs[self.monitor][-1])
        self.y_val.append(val_logs[self.monitor][-1])
        self.ax.clear()
        self.ax.set_yscale(self.scale)
        self.ax.plot(self.x, self.y_train, 'b-', self.x, self.y_val, 'g-')
        self.fig.canvas.draw()
        # plt.pause(0.5)
        return
