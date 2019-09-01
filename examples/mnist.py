import _pickle as cPickle
import gzip
import wget

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pyjet.models import SLModel
from pyjet.data import NpDataset
import pyjet.backend as J
from pyjet.layers import Conv2D, MaxPooling2D, FullyConnected, Input
from pyjet.callbacks import ModelCheckpoint, Plotter

# Load the dataset
try:
    f = gzip.open("mnist_py3k.pkl.gz", "rb")
except OSError:
    print("Could not find MNIST, downloading the dataset")
    wget.download("http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz")
    f = gzip.open("mnist_py3k.pkl.gz", "rb")
(xtr, ytr), (xval, yval), (xte, yte) = cPickle.load(f)
# Need to convert to keras format
f.close()

xtr = xtr.reshape((-1, 1, 28, 28))  # Should be (Height, Width, Channel)
xval = xval.reshape((-1, 1, 28, 28))  # Should be (Height, Width, Channel)
xte = xte.reshape((-1, 1, 28, 28))  # Should be (Height, Width, Channel)

print("Maximum Pixel value in training set:", np.max(xtr))
print("Training Data Shape:", xtr.shape)
print("Training Labels Shape:", ytr.shape)
print("Validation Data Shape: ", xval.shape)
print("Validation Labels Shape: ", yval.shape)

# Visualize an image
ind = np.random.randint(xtr.shape[0])
# plt.imshow(xtr[ind, 0, :, :], cmap='gray')
# plt.title("Digit = %s" % ytr[ind])
# plt.show()


# Create the model
class MNISTModel(SLModel):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = Conv2D(20, kernel_size=5, activation="sigmoid")
        self.conv2 = Conv2D(30, kernel_size=5, activation="sigmoid")
        self.mp = MaxPooling2D(2)
        self.fc1 = FullyConnected(50, activation="linear")
        self.fc2 = FullyConnected(10)

        self.infer_inputs(Input(1, 28, 28))

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp(x)

        x = self.conv2(x)
        x = self.mp(x)

        x = J.flatten(x)

        x = self.fc1(x)
        self.loss_in = self.fc2(x)

        return F.softmax(self.loss_in, dim=-1)


model = MNISTModel()
model.add_loss(nn.CrossEntropyLoss())

# This will save the best scoring model weights to the current directory
best_model = ModelCheckpoint(
    "mnist_pyjet" + ".state",
    monitor="val_accuracy",
    mode="max",
    verbose=1,
    save_best_only=True,
)
# This will plot the model's accuracy during training
plotter = Plotter(scale="linear", monitor="accuracy")

# Turn the numpy dataset into a BatchGenerator
train_datagen = NpDataset(xtr, y=ytr).flow(batch_size=64, shuffle=True, seed=1234)
# Turn the val data into a BatchGenerator
val_datagen = NpDataset(xval, y=yval).flow(batch_size=1000, shuffle=True, seed=1234)

# Set up the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
model.add_optimizer(optimizer)

# Fit the model
model.fit_generator(
    train_datagen,
    epochs=10,
    steps_per_epoch=train_datagen.steps_per_epoch,
    validation_data=val_datagen,
    validation_steps=val_datagen.steps_per_epoch,
    metrics=["accuracy", "top3_accuracy"],
    callbacks=[best_model, plotter],
)

# Load the best model
model = MNISTModel()
print("Loading the model")
model.load_state("mnist_pyjet.state")
# Test it on the test set
test_datagen = NpDataset(xte).flow(batch_size=1000, shuffle=False)
model.reset_stats()
test_preds = model.predict_generator(test_datagen, test_datagen.steps_per_epoch)
num_test = xte.shape[0]


# Visualize an image and its prediction
while True:
    ind = np.random.randint(xte.shape[0])
    plt.imshow(xte[ind, 0, :, :], cmap="gray")
    test_pred = test_preds[ind]
    plt.title("Prediction = %s" % np.argmax(test_pred))
    plt.show()
