from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from . import backend as J
from .utils import ProgBar, log


# TODO Not sure whether I'll need to seperate RL models and SL models. Hopefully
# I planned this out right
class SLModel(nn.Module):

    def __init__(self):
        super(SLModel, self).__init__()
        self.logs = defaultdict(list)

    def forward(*inputs, **kwargs):
        raise NotImplementedError

    def to_torch_x(self, x):
        return Variable(J.Tensor(x.astype(float)))

    def to_torch_y(self, y):
        return self.to_torch_x(y)

    def to_numpy_pred(self, preds):
        return preds.data.cpu().numpy()

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
        numpy_preds = self.to_numpy_pred(preds)
        # Clean torch_x, preds, and torch_y from memory
        # torch_x, preds, torch_y = None, None, None
        # Calculate metrics if we were given any using Numpy
        metric_scores = [metric(numpy_preds, y) for metric in metrics]
        # loss = 0
        # metric_scores = [0 for metric in metrics]
        return loss, metric_scores

    def fit_generator(self, datagen, epochs, steps_per_epoch, optimizer, loss_fn,
                      validation_data=None, val_steps=None,
                      metrics=tuple([]), verbose=1, initial_epoch=0):
        # Run each epoch
        for ep_num in range(initial_epoch, epochs):
            log("Epoch %s/%s" % (ep_num + 1, epochs), verbose, 1)
            # Set up the progress bar
            progbar = ProgBar(verbosity=verbose)
            # Run each step of the epoch
            for step in progbar(steps_per_epoch):
                # TODO create a datagen enqueuer
                x, y = next(datagen)
                batch_loss, batch_scores = self.train_on_batch(x, y, optimizer, loss_fn, metrics)
                # Update the training loss and metrics
                progbar.update("loss", batch_loss.data[0])
                for m_ind, metric in enumerate(metrics):
                    progbar.update(metric.__name__, batch_scores[m_ind])
            # BUG Validation stats do not show up
            if validation_data is not None:
                # Run the validation
                val_loss, val_metrics = self.evaluate_generator(validation_data, val_steps, loss_fn=loss_fn, metrics=metrics, verbose=0)
                # Update the postfix
                log("\tval_loss: %s" % val_loss, verbose, 1)
                for m_ind, metric in enumerate(metrics):
                    log("\tval_" + metric.__name__ + ": %s" % val_metrics[m_ind], verbose, 1)
            # TODO replace this with callbacks
            # Log the loss and metrics
            for key in progbar.postfix.keys():
                self.logs[key].append(progbar.postfix[key])

    def evaluate_on_batch(self, x, y, loss_fn=None, metrics=tuple([])):
        torch_x = self.to_torch_x(x)
        preds = self(torch_x)
        torch_y = self.to_torch_y(y)
        # Calculate the loss
        # TODO thread this so the loss and metrics computation happen in parallel
        loss = loss_fn(preds, torch_y) if loss_fn is not None else None
        # Calculate metrics if we were given any using Numpy
        # Undo the logsoftmax on cpu
        numpy_preds = self.to_numpy_pred(preds)
        metric_scores = [metric(numpy_preds, y) for metric in metrics]
        return loss, metric_scores

    def evaluate_generator(self, datagen, steps, loss_fn=None, metrics=tuple([]), verbose=0):
        # Set up the progress bar
        progbar = ProgBar(verbosity=verbose)
        for step in progbar(steps):
            x, y = next(datagen)
            batch_loss, batch_scores = self.evaluate_on_batch(x, y, loss_fn, metrics)
            # Update the training loss and metrics
            # TODO Very hacky, fix this
            if batch_loss is not None:
                progbar.update("val_loss", batch_loss.data[0])
            else:
                progbar.postfix["val_loss"] = None
            for m_ind, metric in enumerate(metrics):
                progbar.update("val_" + metric.__name__, batch_scores[m_ind])
        return batch_loss.data[0], batch_scores

    def predict_on_batch(self, x):
        self.eval()
        torch_x = self.to_torch_x(x)
        preds = self(torch_x)
        self.train()
        return self.to_numpy_pred(preds)

    def predict_generator(self, datagen, steps, verbose=0):
        # Set up the progress bar
        progbar = ProgBar(verbosity=verbose)
        all_preds = []
        for step in progbar(steps):
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
