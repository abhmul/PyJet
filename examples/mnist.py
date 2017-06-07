import _pickle as cPickle
import gzip, numpy
import wget

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pyjet.models import SLModel
from pyjet.data import NpDataset, DatasetGenerator
import pyjet.backend as J
from pyjet.losses import categorical_crossentropy
from pyjet.metrics import accuracy
from pyjet.utils import to_categorical

# Load the dataset
try:
    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
except:
    print("Could not find MNIST, downloading the dataset")
    wget.download("http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz")
    f = gzip.open('mnist_py3k.pkl.gz', 'rb')
(xtr, ytr), (xval, yval), (xte, yte) = cPickle.load(f)
# Need to convert to keras format
f.close()

ytr = to_categorical(ytr)
yval = to_categorical(yval)
xtr = xtr.reshape((-1, 1,  28, 28)) # Should be (Channel Height, Width)
xval = xval.reshape((-1, 1,  28, 28)) # Should be (Channel Height, Width)

print("Training Data Shape: ", xtr.shape)
print("Training Labels Shape: ", ytr.shape)
print("Validation Data Shape: ", xval.shape)
print("Validation Labels Shape: ", yval.shape)

# Visualize an image
ind = np.random.randint(xtr.shape[0])
plt.imshow(xtr[ind, 0, :, :], cmap='gray')
plt.title("Digit = %s" % np.where(ytr[ind] == 1)[0][0])
plt.show()

# Create the model
class MNISTModel(SLModel):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Set up the weights
        # Make wrappers for this
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5).cuda()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5).cuda()
        self.conv2_drop = nn.Dropout2d().cuda()
        self.fc1 = nn.Linear(320, 50).cuda()
        self.fc2 = nn.Linear(50, 10).cuda()

    def forward(self, x):
        # Define the neural net forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = J.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

model = MNISTModel()

# Turn the numpy dataset into a BatchGenerator
train_datagen = DatasetGenerator(NpDataset(xtr, y=ytr), batch_size=64, shuffle=True, seed=1234)
# Turn the val data into a BatchGenerator
val_datagen = DatasetGenerator(NpDataset(xval, y=yval), batch_size=1000, shuffle=True, seed=1234)

# Set up the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Fit the model
model.fit_generator(train_datagen, epochs=30,
                    steps_per_epoch=train_datagen.steps_per_epoch,
                    optimizer=optimizer,
                    loss_fn=categorical_crossentropy, validation_data=val_datagen,
                    val_steps=val_datagen.steps_per_epoch, metrics=[accuracy],
                    verbose=1)
