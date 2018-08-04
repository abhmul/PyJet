import warnings
import itertools
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable

from . import backend as J
from .training import TrainingLogs, LossManager, OptimizerManager
from .metrics import Metric, AverageMetric
from .callbacks import ProgressBar, CallbackList
from .registry import load_metric

python_iterables = {list, set, tuple, frozenset}


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, itertools.chain([first], iterable)


def standardize_list_input(inputs):
    if type(inputs) in python_iterables:
        return list(inputs)
    return [inputs]


def standardize_metric_input(metrics):
    old_metrics = standardize_list_input(metrics)
    metrics = []
    for metric in old_metrics:
        if isinstance(metric, str):
            metrics.append(load_metric(metric))
        elif isinstance(metric, Metric):
            metrics.append(metric)
        else:
            metrics.append(AverageMetric(metric))
    return metrics


# TODO Not sure whether I'll need to seperate RL models and SL models.
# Hopefully I planned this out right
class SLModel(nn.Module):
    def __init__(self, torch_module=None):
        super(SLModel, self).__init__()
        self.to_cuda = J.use_cuda
        self.loss_in = []
        self.torch_module = torch_module

        self.loss_manager = LossManager()
        self.optimizer_manager = OptimizerManager()

    def infer_inputs(self, *inputs, **kwargs):
        with torch.no_grad():
            self.forward(*inputs, **kwargs)

    def parameters(self, *args, **kwargs):
        params = super(SLModel, self).parameters(*args, **kwargs)
        param_peek = peek(params)
        if param_peek is None:
            warnings.warn("Model has no parameters! Did you forget to call "
                          "infer_inputs?")
            return []
        return param_peek[1]

    def forward(self, *inputs, **kwargs):
        if self.torch_module is not None:
            self.loss_in = self.torch_module.forward(*inputs, **kwargs)
            return self.loss_in
        raise NotImplementedError()

    def cast_input_to_torch(self, x):
        return Variable(J.from_numpy(x))

    def cast_target_to_torch(self, y):
        return Variable(J.from_numpy(y))

    def cast_output_to_numpy(self, preds):
        return preds.data.cpu().numpy()

    def cast_model_to_cuda(self):
        if self.to_cuda:
            self.cuda()
            self.to_cuda = False
        return

    def add_optimizer(self, optimizer, name=None):
        self.optimizer_manager.add_optimizer(optimizer, name=name)

    def remove_optimizer(self, name=None):
        return self.optimizer_manager.remove_optimizer(name=name)

    def clear_optimizers(self):
        self.optimizer_manager.clear_optimizers()

    def loss(self, targets):
        return self.loss_manager.loss(self, targets)

    def add_loss(self, loss_fn, inputs=(), weight=1.0, name=None):
        inputs = standardize_list_input(inputs)
        # Use 'loss_in' if no inputs provided
        if not len(inputs):
            inputs = ['loss_in']
        return self.loss_manager.add_loss(loss_fn,
                                          inputs,
                                          weight=weight,
                                          name=name)

    def remove_loss(self, name=None):
        return self.loss_manager.remove_loss(name=name)

    def clear_losses(self):
        self.loss_manager.clear_losses()

    def compile_loss(self, loss_fn=None):
        """
        This is a function to standardize loss input and hack it to behave like
        a metric. A few key notes to remember:
            - If the loss_fn is None, it will just use the loss method
              defined by the model. This by default comes from the loss manager
              which is modified by the add_loss, remove_loss, and clear_losses
              methods. If a loss_fn is provided, then this method will clear
              all current losses from the loss manager and add the input loss
              function to it, taking as input the default "loss_in" parameter.
              If you override the model's loss function, then passing a loss_fn
              will have no effect!
            - If there is more than one loss in the loss manager, then this
              function will also return metric versions of all the auxilary
              losses. The overall loss function is only computed once,
              the auxilary loss scores are taken from loss cache.

        Args:
            loss_fn: The loss function to compile. Defaults to None. See above
                note for explanation of behavior when None and when not None.

        Returns:
            (tuple): All the relevant loss functions in a tuple. See above note
                for more explanation about how this return value is determined
        """
        # if loss_fn is defined, clear the losses, and set it to the input
        # loss_fn
        if loss_fn is not None:
            if len(self.loss_manager):
                warnings.warn("Loss manager is not empty, but loss_fn passed "
                              "passed to fit_generator or validate_generator."
                              " Clearing all past losses.")
                self.clear_losses()
                self.add_loss(loss_fn)

        # Compile the main loss
        def loss(preds, targets):
            # Preds are not used, just works as metric)
            return self.loss(targets)

        # Compile the auxilary losses, the main loss must be called before
        # the auxilary losses
        aux_losses = []
        # Only account for auxilary losses if there is more than one loss
        if len(self.loss_manager) > 1:
            for name in self.loss_manager.names:
                # Using the default value gets around the problem of late
                # binding.
                # https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
                def aux_loss(preds, targets, name=name):
                    # Preds are not used, just hack to make it behave like
                    # metric
                    return self.loss_manager.get_loss_score(name=name)
                metric_aux_loss = AverageMetric(aux_loss)
                # Change the name for logging
                metric_aux_loss.__name__ = name
                aux_losses.append(metric_aux_loss)

        return (AverageMetric(loss), *aux_losses)

    def train_on_batch(self, x, target, optimizers, loss_fn, metrics=()):
        """
        Trains the SLModel on a single batch of data.

        Args:
            x: A batch of input into the model.
            target: The corresponding labels for the batch x.
            optimizers: A list of optimizers to run with the model.
            loss_fn: The loss function to run on the model
            metrics: A list of metrics to calculate on the output of the model

        Returns:
            A tuple where the first element is the loss over the batch and the
            second element is a list of the scores corresponding to the input
            metrics.
        """
        self.cast_model_to_cuda()
        self.train()
        # Cast inputs to a torch variable
        torch_x = self.cast_input_to_torch(x)
        torch_target = self.cast_target_to_torch(target)
        # Make the prediction
        torch_preds = self(torch_x)
        # Calculate the loss
        loss = loss_fn(torch_preds, torch_target)
        # Update the weights
        [optimizer.zero_grad() for optimizer in optimizers]
        loss.backward()
        [optimizer.step() for optimizer in optimizers]
        # Calculate the metrics
        metric_scores = [
            metric(torch_preds, torch_target) for metric in metrics
        ]
        # Clean up some variables
        self.zero_grad()
        del torch_x
        del torch_target
        del torch_preds
        if J.use_cuda:
            torch.cuda.empty_cache()
        return loss, metric_scores

    def validate_on_batch(self, x, target, metrics):
        self.cast_model_to_cuda()
        self.eval()
        with torch.no_grad():
            # Cast inputs to a torch variable and set to volatile for inference
            torch_x = self.cast_input_to_torch(x)
            torch_target = self.cast_target_to_torch(target)
            # Make the prediction
            torch_preds = self(torch_x)
            preds = self.cast_output_to_numpy(torch_preds)
            # Calculate the metrics
            metric_scores = [
                metric(torch_preds, torch_target) for metric in metrics
            ]
            # Clean up some variables
            del torch_x
            del torch_preds
            del torch_target
            if J.use_cuda:
                torch.cuda.empty_cache()
        return metric_scores, preds

    def validate_generator(self,
                           val_generator,
                           validation_steps,
                           loss_fn=None,
                           metrics=(),
                           verbose=0):
        self.cast_model_to_cuda()
        metrics = standardize_metric_input(metrics)
        if loss_fn is not None or len(self.loss_manager):
            loss_fn, *aux_loss_fns = self.compile_loss(loss_fn)
            metrics = [loss_fn] + metrics + aux_loss_fns
        # Set up the logs
        logs = TrainingLogs()
        # Set the model to eval mode
        self.eval()
        callbacks = [ProgressBar(validation_steps)] if verbose > 0 else []
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin(logs=logs)
        callbacks.on_epoch_begin(0, logs=logs.epoch_logs)
        for step in range(validation_steps):
            callbacks.on_batch_begin(epoch=0, step=step, logs=logs.batch_logs)
            x, target = next(val_generator)
            b_metrics, _ = self.validate_on_batch(x, target, metrics)
            for metric, score in zip(metrics, b_metrics):
                logs.log_metric(metric, score)
            callbacks.on_batch_end(epoch=0, step=step, logs=logs.batch_logs)
        callbacks.on_epoch_end(0, logs=logs.epoch_logs)
        callbacks.on_train_end(logs=logs)
        return logs.epoch_logs

    def fit_generator(self,
                      generator,
                      steps_per_epoch,
                      epochs,
                      validation_data=None,
                      validation_steps=0,
                      metrics=(),
                      callbacks=(),
                      initial_epoch=0,
                      verbose=1):
        self.cast_model_to_cuda()
        # Standardize the input
        optimizers = self.optimizer_manager.optimizers
        loss_fn, *aux_loss_fns = self.compile_loss()
        metrics = standardize_metric_input(metrics) + aux_loss_fns
        callbacks = CallbackList(callbacks)
        # If the verbosity is set, set up the progress bar
        if verbose > 0:
            callbacks.append(ProgressBar(steps_per_epoch, epochs=epochs))
        # Register the model with each callback
        callbacks.set_model(self)
        # Save whether we will need to run validation
        run_validation = (validation_steps >
                          0) and validation_data is not None
        logs = TrainingLogs()

        # Run the callbacks
        callbacks.on_train_begin(logs=logs)
        # Loop through all the epochs
        for epoch in range(initial_epoch, epochs):
            # Put the model in train mode
            self.train()
            # Reset the metrics
            loss_fn = loss_fn.reset()
            metrics = [metric.reset() for metric in metrics]
            # Run the callbacks
            logs.on_epoch_begin()
            callbacks.on_epoch_begin(epoch, logs=logs.epoch_logs)
            # Run each step of the epoch with a progress bar
            for step in range(steps_per_epoch):
                # Run the callbacks
                callbacks.on_batch_begin(
                    epoch=epoch, step=step, logs=logs.batch_logs)
                x, target = next(generator)
                b_loss, b_metrics = self.train_on_batch(
                    x, target, optimizers, loss_fn, metrics)
                # Add stats to the logs
                logs.log_metric(loss_fn, b_loss)
                for score, metric in zip(b_metrics, metrics):
                    logs.log_metric(metric, score)
                # Run the callbacks
                callbacks.on_batch_end(
                    epoch=epoch, step=step, logs=logs.batch_logs)

            # Check if we need to run validation
            if run_validation:
                loss_fn = loss_fn.reset()
                metrics = [metric.reset() for metric in metrics]
                self.validate_generator(
                    validation_data,
                    validation_steps,
                    metrics=([loss_fn] + metrics))
                # Log the loss and metrics
                for metric in [loss_fn] + metrics:
                    logs.log_validation_metric(metric)
            # Run the callbacks
            logs.on_epoch_end()
            callbacks.on_epoch_end(epoch, logs=logs.epoch_logs)
        # Run the callbacks
        callbacks.on_train_end(logs=logs)
        # Put the model back in eval mode
        self.eval()
        return logs

    def predict_on_batch(self, x):
        self.cast_model_to_cuda()
        self.eval()
        with torch.no_grad():
            # Cast inputs to a torch variable and set to volatile for inference
            torch_x = self.cast_input_to_torch(x)
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
        progbar = tqdm if verbose > 0 else lambda x: x
        for _ in progbar(range(prediction_steps)):
            x = next(generator)
            batch_preds = self.predict_on_batch(x)
            # Check to make sure the ndim is the same
            if len(preds) > 0:
                assert batch_preds.ndim == preds[-1].ndim
            preds.append(batch_preds)

        # Supports variable sized predictions - get the biggest possible shape
        num_preds = sum(len(batch_preds) for batch_preds in preds)
        max_shape = [num_preds] + [
            max(preds[n].shape[i] for n in range(len(preds)))
            for i in range(1, preds[0].ndim)
        ]
        full_preds = np.zeros(max_shape, dtype=preds[0].dtype)
        # Fill in the predictions array
        cur_pred_ind = 0
        for batch_preds in preds:
            preds_slice = (slice(cur_pred_ind,
                                 cur_pred_ind + len(batch_preds)), ) + tuple(
                                     slice(batch_preds.shape[i])
                                     for i in range(1, batch_preds.ndim))
            full_preds[preds_slice] = batch_preds
            cur_pred_ind += len(batch_preds)

        return full_preds

    def save_state(self, save_path):
        return torch.save(self.state_dict(), save_path)

    def load_state(self, load_path):
        self.load_state_dict(torch.load(load_path))
