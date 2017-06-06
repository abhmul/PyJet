import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

import backend as J
import numpy as np
from utils import ProgBar, log


# TODO Not sure whether I'll need to seperate RL models and SL models. Hopefully
# I planned this out right
class SLModel(nn.Module):

    def forward(*inputs, **kwargs):
        raise NotImplementedError

    def to_torch_x(self, x):
        return Variable(J.Tensor(x))

    def to_torch_y(self, y):
        return self.to_torch_x(y)

    def to_numpy_pred(self, preds):
        return preds.numpy()

    # TODO add in basic metrics (accuracy, top-k, etc.)
    def train_on_batch(self, x, y, optimizer, loss_fn, metrics=tuple([])):
        torch_x = self.to_torch_x(x)
        preds = self(torch_x)
        torch_y = self.to_torch_y(y)
        # Calculate the loss
        # TODO thread this so the loss and metrics computation happen in parallel
        loss = loss_fn(preds, torch_y)
        # Update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Undo the logsoftmax on cpu
        # TODO Fix the logsoftmax problem
        numpy_preds = self.to_numpy_pred(preds)
        if isinstance(numpy_preds, np.ndarray):
            numpy_preds = np.exp(numpy_preds)
        else:
            numpy_preds = [np.exp(numpy_pred) for numpy_pred in numpy_preds]
        # Calculate metrics if we were given any using Numpy
        metric_scores = [metric(numpy_preds, y) for metric in metrics]
        return loss, metric_scores

    def fit_generator(self, datagen, epochs, steps_per_epoch, optiimizer, loss_fn,
                      validation_data=None, val_steps=None,
                      metrics=tuple([]), verbose=1, initial_epoch=0):
        # Run each epoch
        for ep_num in range(initial_epoch, epochs):
            log("Epoch %s/%s" % (ep_num + 1, epochs), verbose, 1)
            # Set up the progress bar
            progbar = ProgBar(verbosity=verbose)
            # Run each step of the epoch
            for step in progbar(range(steps_per_epoch)):
                # TODO create a datagen enqueuer
                x, y = next(datagen)
                batch_loss, batch_scores = self.train_on_batch(x, y, optimizer, loss_fn, metrics)
                # Update the training loss and metrics
                progbar.update("loss", batch_loss)
                for m_ind, metric in enumerate(metrics):
                    progbar.update(metric.__name__, batch_scores[m_ind])
            if validation_data is not None:
                # Run the validation
                val_loss, val_metrics = self.evaluate_generator(validation_data, val_steps, loss_fn=loss_fn, metrics=metrics, verbose=0)
                # Update the postfix
                progbar.update("val_loss", val_loss)
                for m_ind, metric in enumerate(metrics):
                    progbar.update("val_" + metric.__name__, val_metrics[m_ind])
            # TODO replace this with callbacks
            # Log the loss and metrics
            for key in postfix.keys():
                self.logs[key].append(postfix[key])

    def evaluate_on_batch(self, x, y, loss_fn=None, metrics=tuple([])):
        torch_x = self.to_torch_x(x)
        preds = self(torch_x, mode='train')
        torch_y = self.to_torch_y(y)
        # Calculate the loss
        # TODO thread this so the loss and metrics computation happen in parallel
        loss = loss_fn(preds, torch_y) if loss_fn is not None else None
        # Calculate metrics if we were given any using Numpy
        # Undo the logsoftmax on cpu
        numpy_preds = self.to_numpy_pred(preds)
        if isinstance(numpy_preds, np.ndarray):
            numpy_preds = np.exp(numpy_preds)
        else:
            numpy_preds = [np.exp(numpy_pred) for numpy_pred in numpy_preds]
        metric_scores = [metric(preds, y_labels) for metric in metrics]
        return loss, metric_scores

    def evaluate_generator(self, datagen, steps, loss_fn=None, metrics=tuple([]), verbose=0):
        # Set up the progress bar
        progbar = ProgBar(verbosity=verbose)
        for step in progbar(range(steps)):
            x, y = next(datagen)
            batch_loss, batch_scores = self.evaluate_on_batch(x, y, loss_fn, metrics)
            # Update the training loss and metrics
            progbar.update("val_loss", batch_loss)
            for m_ind, metric in enumerate(metrics):
                progbar.update("val_" + metric.__name__, batch_scores[m_ind])
        return progbar.postfix["val_loss"], [progbar.postfix["val_" + metric.__name__] for metric in metrics]

    def predict_on_batch(self, x):
        torch_x = self.to_torch_x(x)
        preds = self(torch_x, mode='test')
        return self.to_numpy_pred(preds)

    def predict_generator(self, datagen, steps, verbose=0):
        # Set up the progress bar
        progbar = ProgBar(verbosity=verbose)
        all_preds = []
        for step in progbar(range(steps)):
            x = next(datagen)
            all_preds.append(self.predict_on_batch(x))
        # Try to stack, but just return the list if we can't
        try:
            return np.stack(all_preds)
        except AttributeError:
            return all_preds

    def save_state(self, save_path):
        return torch.save(self.state_dict(), save_path)

    def load_state(self, load_path):
        self.load_state_dict(torch.load(load_path))
