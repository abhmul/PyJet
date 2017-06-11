from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import _pickle as cPickle
import gzip, numpy
import wget
import numpy as np
import matplotlib.pyplot as plt
from pyjet.utils import to_categorical
from pyjet.data import NpDataset, DatasetGenerator

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


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

xtr = xtr.reshape((-1, 1,  28, 28)) # Should be (Channel Height, Width)
xval = xval.reshape((-1, 1,  28, 28)) # Should be (Channel Height, Width)

print(np.max(xtr))
print("Training Data Shape: ", xtr.shape)
print("Training Labels Shape: ", ytr.shape)
print("Validation Data Shape: ", xval.shape)
print("Validation Labels Shape: ", yval.shape)

# Visualize an image
ind = np.random.randint(xtr.shape[0])
plt.imshow(xtr[ind, 0, :, :], cmap='gray')
plt.title("Digit = %s" % ytr[ind])
plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return J.softmax(x)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Turn the numpy dataset into a BatchGenerator
train_datagen = DatasetGenerator(NpDataset(xtr, y=ytr), batch_size=32, shuffle=True, seed=1234)
# Turn the val data into a BatchGenerator
val_datagen = DatasetGenerator(NpDataset(xval, y=yval), batch_size=1000, shuffle=True, seed=1234)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_datagen):
        if args.cuda:
            data, target = torch.Tensor(data).cuda(), torch.LongTensor(target).cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(xtr),
                100. * batch_idx / train_datagen.steps_per_epoch, loss.data[0]))
        if train_datagen.steps_per_epoch == batch_idx + 1:
            break

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(val_datagen):
        if args.cuda:
            data, target = torch.Tensor(data).cuda(), torch.LongTensor(target).cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        if val_datagen.steps_per_epoch == batch_idx + 1:
            break

    test_loss = test_loss
    test_loss /= val_datagen.steps_per_epoch # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(xval),
        100. * correct / len(xval)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test(epoch)
