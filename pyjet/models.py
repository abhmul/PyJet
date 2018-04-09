import warnings
from functools import wraps

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import backend as J
from .training import ProgBar, BatchLogs
from .callbacks import CallbackList


python_iterables = {list, set, tuple, frozenset}


def standardize_list_input(inputs):
    if type(inputs) in python_iterables:
        return list(inputs)
    return [inputs]

# TODO: Fix these decorators
def loss_function(f):

    f.__annotations__["loss"] = True
    f.__annotations__["function"] = f
    def wrapper(*args):
        output = f(*args)
        return output

    return wrapper


def metric_function(metric):

    metric.__annotations__["metric"] = True
    metric.__annotations__["function"] = metric
    def wrapper(*args):
        output = metric(*args)
        return output

    return wrapper


def load_model(load_path):
    return torch.load(load_path)


# TODO Not sure whether I'll need to seperate RL models and SL models. Hopefully
# I planned this out right
class SLModel(nn.Module):

    def __init__(self, torch_module=None):
        super(SLModel, self).__init__()
        self.to_cuda = J.use_cuda
        self.torch_module = torch_module
        self._loss_functions = {}
        self._metric_functions = {}
        self._optimizers = {}
        self._cache = {}
        self.stop_training = False

        # # Check if we have any loss functions or metric functions and register them
        # for attr in self.__dict__.values():
        #     try:
        #         if attr.__annotations__.get("loss"):
        #             self.register_loss_function(attr.__annotations__["function"])
        #         if attr.__annotations__.get("metric"):
        #             self.register_metric_function(attr.__annotations__["function"])
        #     except:
        #         pass

    def call(self, *inputs, **kwargs):
        if self.torch_module is not None:
            return self.torch_module(*inputs, **kwargs)
        raise NotImplementedError("Need to implement the call function to use this model.")

    def forward(self, *inputs, **kwargs):
        self._cache["output"] = self.call(*inputs, **kwargs)
        return self._cache["output"]

    def cast_input_to_torch(self, x, volatile=False):
        return Variable(J.from_numpy(x), volatile=volatile)

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y), volatile=volatile)

    def cast_output_to_numpy(self, preds):
        return J.to_numpy(preds.data)

    def cast_model_to_cuda(self):
        if self.to_cuda:
            self.cuda()
            self.to_cuda = False
        return

    def add_optimizer(self, optimizer, name=None):
        if name is None:
            name = optimizer.__class__.__name__
        if name in self._optimizers:
            raise ValueError("Optimizer with name {} already added".format(name))
        self._optimizers[name] = optimizer

    def remove_optimizer(self, name=None, optimizer=None):
        if name is None:
            assert optimizer is not None, "Must either pass a name or optimizer to remove."
            name = optimizer.__class__.__name__
        self._optimizers.pop(name)

    def add_loss_function(self, loss_fn, name=None):
        if name is None:
            name = loss_fn.__name__

        # Create the loss function
        def new_loss_function(targets):
            return loss_fn(self._cache["output"], targets)
        # Change the name
        new_loss_function.__name__ = name
        # Register the loss function
        self.register_loss_function(new_loss_function)

    def add_metric_function(self, metric_function, name=None):
        if name is None:
            name = metric_function.__name__

        # Create the loss function
        def new_metric_function(targets):
            return metric_function(self._cache["output"].data, targets.data)[0]
        # Change the name
        new_metric_function.__name__ = name
        # Register the loss function
        self.register_metric_function(new_metric_function)

    def register_loss_function(self, loss):
        print("Calling register on %s" % loss.__name__)
        if loss.__name__ in self._metric_functions:
            raise ValueError("Cannot have a loss with the same name as metric {}".format(loss.__name__))
        self._loss_functions[loss.__name__] = loss

    def remove_loss_function(self, loss):
        self._metric_functions.pop(loss.__name__)

    def register_metric_function(self, metric):
        if metric.__name__ in self._loss_functions:
            raise ValueError("Cannot have a loss with the same name as metric {}".format(metric.__name__))
        self._metric_functions[metric.__name__] = metric

    def remove_metric_function(self, metric):
        self._metric_functions.pop(metric.__name__)

    def loss(self, targets):
        output_dict = {loss_name: loss_fn( targets) for loss_name, loss_fn in self._loss_functions.items()}
        if not output_dict:
            return Variable(J.zeros(1)), {}
        return sum(output_dict.values()), output_dict

    def run_metrics(self, targets):
        output_dict = {metric_name: metric_fn(targets) for metric_name, metric_fn in
                       self._metric_functions.items()}
        return output_dict

    @staticmethod
    def combine_output_scores(loss, loss_dict, metric_dict):
        # Get the float value of the loss
        output_dict = {}
        if len(loss_dict) > 1:
            output_dict = {loss_name: loss_value.data[0] for loss_name, loss_value in loss_dict.items()}
        # Include the total loss value
        output_dict.update(loss=loss.data[0])
        # Include the metrics in the output dict
        output_dict.update(metric_dict)
        return output_dict

    def make_batch_logs(self):
        return BatchLogs("loss",
                         *(self._loss_functions.keys() if len(self._loss_functions) > 1 else []),
                         *self._metric_functions.keys())

    def make_training_logs(self, run_validation=False):
        names = ["loss", *(self._loss_functions.keys() if len(self._loss_functions) > 1 else []),
                 *self._metric_functions.keys()]
        if run_validation:
            names += ["val_" + name for name in names]
        return BatchLogs(names)

    def train_on_batch(self, x, target):
        self.cast_model_to_cuda()
        self.train()
        # Cast inputs to a torch variable
        torch_x = self.cast_input_to_torch(x)
        torch_target = self.cast_target_to_torch(target)
        # Make the prediction
        torch_preds = self(torch_x)
        # Calculate the loss
        loss, loss_dict = self.loss(torch_target)
        # Update the weights
        [optimizer.zero_grad() for optimizer in self._optimizers]
        loss.backward()
        [optimizer.step() for optimizer in self._optimizers]
        # Calculate the metrics
        metric_scores = self.run_metrics(torch_target)
        # Reset the optimizer
        self.zero_grad()
        # Clean up some variables
        del torch_x
        del torch_target
        del torch_preds
        if J.use_cuda:
            torch.cuda.empty_cache()

        # Making the logs
        return self.combine_output_scores(loss, loss_dict, metric_scores)

    def test_on_batch(self, x, target):
        self.cast_model_to_cuda()
        self.eval()
        # Cast inputs to a torch variable and set to volatile for inference
        torch_x = self.cast_input_to_torch(x, volatile=True)
        torch_target = self.cast_target_to_torch(target, volatile=True)
        # Make the prediction
        torch_preds = self(torch_x)
        # Calculate the metrics
        loss, loss_dict = self.loss(torch_target)
        metric_scores = self.run_metrics(torch_target)
        # Clean up some variables
        del torch_x
        del torch_preds
        del torch_target
        if J.use_cuda:
            torch.cuda.empty_cache()

        # Making the logs
        return self.combine_output_scores(loss, loss_dict, metric_scores)

    def evaluate_generator(self, generator, steps=None, verbose=0):
        if steps is None:
            steps = len(generator)

        self.cast_model_to_cuda()
        # Set up the logs
        batch_logs = self.make_batch_logs()
        # Set the model to eval mode
        self.eval()
        progbar = ProgBar(verbosity=verbose)
        for step in progbar(steps):
            x, target = next(generator)
            batch_scores = self.test_on_batch(x, target)
            batch_logs.update(batch_scores)
        return batch_logs.average()

    def fit_generator(self, generator, steps_per_epoch, epochs, callbacks=None,
                      validation_data=None, validation_steps=0, initial_epoch=0,
                      verbose=1):
        self.cast_model_to_cuda()
        # Register the model with each callback
        callbacks = CallbackList(callbacks)
        # Save whether we will need to run validation
        run_validation = (validation_steps > 0) and validation_data is not None
        # Set up the logs
        training_logs = self.make_training_logs(run_validation=run_validation)
        # Run the callbacks
        callbacks.on_train_begin()
        # Loop through all the epochs
        for epoch in range(initial_epoch, epochs):
            # Check if we should stop training
            if self.stop_training:
                break
            # Put the model in train mode
            self.train()
            if verbose > 0:
                print("Epoch {curr}/{total}".format(curr=epoch + 1, total=epochs))
            # Setup the progress bar
            progbar = ProgBar(verbosity=verbose)
            # Setup the batch logs
            batch_logs = self.make_batch_logs()
            # Run the callbacks
            callbacks.on_epoch_begin(epoch)
            # Run each step of the epoch with a progress bar
            for step in progbar(steps_per_epoch):
                x, target = next(generator)
                # Run the callbacks
                callbacks.on_batch_begin(step, {"size": len(x)})
                batch_dict = self.train_on_batch(x, target)
                # Add stats to the batch_logs
                batch_logs.update(batch_dict)
                # Add the stats to the progress bar
                progbar.update_stats(batch_dict)
                progbar.update_bar()
                # Run the callbacks
                callbacks.on_batch_end(step, batch_logs.last_logs)

            # Compute the average stats
            training_logs.update(batch_logs.average())

            # Check if we need to run validation
            if run_validation:
                val_scores = self.validate_generator(validation_data, steps=validation_steps)
                # Append the "val_" to each of the val_metrics
                val_scores = {"val_" + name: value for name, value in val_scores.items()}
                # Update the progress bar
                progbar.update_stats(val_scores)
                progbar.update_bar()
                # Update the training logs
                training_logs.update(val_scores)
            # Run the callbacks
            callbacks.on_epoch_end(epoch, logs=training_logs.last_logs)
        # Run the callbacks
        callbacks.on_train_end()
        # Put the model back in eval mode
        self.eval()
        return training_logs

    def predict_on_batch(self, x):
        self.cast_model_to_cuda()
        self.eval()
        # Cast inputs to a torch variable and set to volatile for inference
        torch_x = self.cast_input_to_torch(x, volatile=True)
        # Make the prediction
        torch_preds = self(torch_x)
        preds = self.cast_output_to_numpy(torch_preds)
        self.zero_grad()
        del torch_x
        del torch_preds
        if J.use_cuda:
            torch.cuda.empty_cache()
        # cast to numpy and return
        return preds

    def predict_generator(self, generator, prediction_steps, verbose=0):
        self.cast_model_to_cuda()
        self.eval()
        preds = []
        # Loop through all the steps
        progbar = ProgBar(verbosity=verbose)
        for step in progbar(prediction_steps):
            x = next(generator)
            batch_preds = self.predict_on_batch(x)
            # Check to make sure the ndim is the same
            assert batch_preds.ndim == preds[-1].ndim
            preds.append(batch_preds)

        # Supports variable sized predictions - get the biggest possible shape
        num_preds = sum(len(batch_preds) for batch_preds in preds)
        max_shape = [num_preds] + [max(preds[n].shape[i] for n in range(len(preds))) for i in range(1, preds[0].ndim)]
        full_preds = np.zeros(max_shape, dtype=preds[0].dtype)
        # Fill in the predictions array
        cur_pred_ind = 0
        for batch_preds in preds:
            preds_slice = (slice(cur_pred_ind, len(batch_preds)),) + tuple(
                slice(batch_preds.shape[i]) for i in range(1, batch_preds.ndim))
            full_preds[preds_slice] = batch_preds
            cur_pred_ind += len(batch_preds)

        return full_preds

    def save_weights(self, save_path, overwrite=True):
        if not overwrite:
            warnings.warn("OVerwrite set to false, but ignoring!")
        return torch.save(self.state_dict(), save_path)

    def load_weights(self, load_path):
        self.load_state_dict(torch.load(load_path))

    def save(self, save_path, overwrite=True):
        if not overwrite:
            warnings.warn("OVerwrite set to false, but ignoring!")
        return torch.save(self, save_path)
