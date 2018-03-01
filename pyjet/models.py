import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from . import backend as J
from .training import ProgBar, MetricLogs


python_iterables = {list, set, tuple, frozenset}


def standardize_list_input(inputs):
    if type(inputs) in python_iterables:
        return list(inputs)
    return [inputs]


# TODO Not sure whether I'll need to seperate RL models and SL models. Hopefully
# I planned this out right
class SLModel(nn.Module):

    def __init__(self, torch_module=None):
        super(SLModel, self).__init__()
        self.to_cuda = J.use_cuda
        self.loss_in = None
        self.aux_loss = []
        self.torch_module = torch_module

    def forward(self, *inputs, **kwargs):
        if self.torch_module is not None:
            self.loss_in = self.torch_module.forward(*inputs, **kwargs)
            return self.loss_in
        raise NotImplementedError()

    def cast_input_to_torch(self, x, volatile=False):
        return Variable(J.from_numpy(x), volatile=volatile)

    def cast_target_to_torch(self, y, volatile=False):
        return Variable(J.from_numpy(y), volatile=volatile)

    def cast_output_to_numpy(self, preds):
        return preds.data.cpu().numpy()

    def cast_model_to_cuda(self):
        if self.to_cuda:
            self.cuda()
            self.to_cuda = False
        return

    def compile_loss(self, loss_fn):
        def loss(preds, targets):
            # Preds are not used, just works as metric)
            return loss_fn(self.loss_in, targets)
        return loss

    def train_on_batch(self, x, target, optimizers, loss_fn, metrics=()):
        self.cast_model_to_cuda()
        self.train()
        # Cast inputs to a torch variable
        torch_x = self.cast_input_to_torch(x)
        torch_target = self.cast_target_to_torch(target)
        # Make the prediction
        torch_preds = self(torch_x)
        # Calculate the loss
        loss = loss_fn(torch_preds, torch_target)
        # Add in the auxilary losses if there are any
        if self.aux_loss:
            loss += sum(aux_loss(torch_target) for aux_loss in self.aux_loss)
        # Update the weights
        [optimizer.zero_grad() for optimizer in optimizers]
        loss.backward()
        [optimizer.step() for optimizer in optimizers]
        # Calculate the metrics
        metric_scores = [metric(torch_preds, torch_target).data[0]
                         for metric in metrics]
        # Clean up some variables
        self.zero_grad()
        loss = loss.data[0]
        del torch_x
        del torch_target
        del torch_preds
        if J.use_cuda:
            torch.cuda.empty_cache()
        return loss, metric_scores

    def validate_on_batch(self, x, target, metrics):
        self.cast_model_to_cuda()
        self.eval()
        # Cast inputs to a torch variable and set to volatile for inference
        torch_x = self.cast_input_to_torch(x, volatile=True)
        torch_target = self.cast_target_to_torch(target, volatile=True)
        # Make the prediction
        torch_preds = self(torch_x)
        preds = self.cast_output_to_numpy(torch_preds)
        # Calculate the metrics
        metric_vals = [metric(torch_preds, torch_target).data[0] for metric in metrics]
        # Clean up some variables
        del torch_x
        del torch_preds
        del torch_target
        if J.use_cuda:
            torch.cuda.empty_cache()
        return metric_vals, preds

    def validate_generator(self, val_generator, validation_steps, loss_fn=None, metrics=(), np_metrics=(),
                           verbose=0):
        self.cast_model_to_cuda()
        metrics = standardize_list_input(metrics)
        np_metrics = standardize_list_input(np_metrics)
        if loss_fn is not None:
            loss_fn = self.compile_loss(loss_fn)
            metrics = [loss_fn, ] + metrics
        # Set up the logs
        batch_logs = MetricLogs(metrics)
        # Set the model to eval mode
        self.eval()
        preds = []
        targets = []
        progbar = ProgBar(verbosity=verbose)
        for step in progbar(validation_steps):
            x, target = next(val_generator)
            b_metrics, b_preds = self.validate_on_batch(x, target, metrics)
            batch_logs.update_logs(metrics, b_metrics)
            if len(np_metrics):
                preds.append(b_preds)
                targets.append(target)
        # Compute the np preds over the whole prediction set
        torch_metric_vals = {metric.__name__: batch_logs.average(metric) for metric in metrics}
        # No-op if np_metrics is not given
        if len(np_metrics):
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
        np_metric_vals = {metric.__name__: metric(preds, targets) for metric in np_metrics}
        return {**torch_metric_vals, **np_metric_vals}

    def fit_generator(self, generator, steps_per_epoch, epochs, optimizer,
                      loss_fn, validation_generator=None, validation_steps=0,
                      metrics=(), np_metrics=(), callbacks=(), initial_epoch=0,
                      verbose=1):
        self.cast_model_to_cuda()
        optimizers = standardize_list_input(optimizer)
        metrics = standardize_list_input(metrics)
        np_metrics = standardize_list_input(np_metrics)
        loss_fn = self.compile_loss(loss_fn)
        # Register the model with each callback
        [callback.set_model(self) for callback in callbacks]
        # Save whether we will need to run validation
        run_validation = (validation_steps >
                          0) and validation_generator is not None
        # Set up the logs\
        train_logs = MetricLogs([loss_fn] + metrics)
        val_logs = MetricLogs([loss_fn] + metrics + np_metrics)
        # Run the callbacks
        [callback.on_train_begin(train_logs=train_logs, val_logs=val_logs)
         for callback in callbacks]
        # Loop through all the epochs
        for epoch in range(initial_epoch, epochs):
            # Put the model in train mode
            self.train()
            if verbose > 0:
                print("Epoch {curr}/{total}".format(curr=epoch + 1, total=epochs))
            # Setup the progress bar
            progbar = ProgBar(verbosity=verbose)
            # Setup the batch logs
            batch_logs = MetricLogs([loss_fn] + metrics)
            # Run the callbacks
            [callback.on_epoch_begin(epoch, train_logs=train_logs, val_logs=val_logs)
             for callback in callbacks]
            # Run each step of the epoch with a progress bar
            for step in progbar(steps_per_epoch):
                # Run the callbacks
                [callback.on_batch_begin(step, train_logs=batch_logs)
                 for callback in callbacks]
                x, target = next(generator)
                # if len(generator_output) == 2:
                # x, target = generator_output
                # else:
                # raise ValueError("Generator output had a length of %s" % len(generator_output))
                b_loss, b_metrics = self.train_on_batch(
                    x, target, optimizers, loss_fn, metrics)
                # Add stats to the batch_logs
                batch_logs.update_logs(
                    [loss_fn] + metrics, [b_loss] + b_metrics)
                # Add the stats to the progress bar
                progbar.update_stats_from_func(
                    [loss_fn] + metrics, [b_loss] + b_metrics)
                progbar.update_bar()
                # Run the callbacks
                [callback.on_batch_end(step, train_logs=batch_logs)
                 for callback in callbacks]

            # Compute the average stats
            for stat in [loss_fn] + metrics:
                train_logs.update_log(stat, batch_logs.average(stat))

            # Check if we need to run validation
            if run_validation:
                val_metrics = self.validate_generator(
                    validation_generator, validation_steps, metrics=([loss_fn] + metrics), np_metrics=np_metrics)
                print("\tValidation Loss: ", val_metrics['loss'])
                for metric_name, metric_score in val_metrics.items():
                    print("\tValidation %s: " % metric_name, metric_score)
                # Log the loss and metrics
                val_metric_funcs = [loss_fn] + metrics + np_metrics
                val_scores = [val_metrics[metric_func.__name__]
                              for metric_func in val_metric_funcs]
                val_logs.update_logs(val_metric_funcs, val_scores)
            # Run the callbacks
            [callback.on_epoch_end(epoch, train_logs=train_logs, val_logs=val_logs)
             for callback in callbacks]
        # Run the callbacks
        [callback.on_train_end(train_logs=train_logs, val_logs=val_logs)
         for callback in callbacks]
        # Put the model back in eval mode
        self.eval()
        return train_logs, val_logs

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
            preds.append(self.predict_on_batch(x))
        return np.concatenate(preds, axis=0)

    def save_state(self, save_path):
        return torch.save(self.state_dict(), save_path)

    def load_state(self, load_path):
        self.load_state_dict(torch.load(load_path))
