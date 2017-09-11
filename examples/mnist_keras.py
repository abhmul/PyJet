import _pickle as cPickle
import gzip
import numpy
import wget
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Dense, Dropout, Input, Activation, MaxPooling2D, Flatten
from keras.optimizers import SGD
from keras.models import Model
from keras.utils import to_categorical


from pyjet.data import NpDataset, DatasetGenerator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

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
xtr = xtr.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)
xval = xval.reshape((-1, 1,  28, 28))  # Should be (Channel Height, Width)

# Change the dimensions for keras/tensorflow
xtr = xtr.transpose(0, 2, 3, 1)
xval = xval.transpose(0, 2, 3, 1)

print("Training Data Shape: ", xtr.shape)
print("Training Labels Shape: ", ytr.shape)
print("Validation Data Shape: ", xval.shape)
print("Validation Labels Shape: ", yval.shape)

# Visualize an image
ind = np.random.randint(xtr.shape[0])
plt.imshow(xtr[ind, :, :, 0], cmap='gray')
plt.title("Digit = %s" % np.where(ytr[ind] == 1)[0][0])
plt.show()

# Create the model


def mnist_model():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(10, (5, 5), padding='valid')(input_img)
    x = MaxPooling2D()(x)
    x = Activation('relu')(x)
    x = Conv2D(20, (5, 5), padding='valid')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Activation('relu')(x)
    x = Dense(50)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    out = Activation('softmax')(x)

    model = Model(inputs=input_img, outputs=out)

    model.compile(SGD(lr=0.01, momentum=0.9, nesterov=False), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = mnist_model()

# Turn the numpy dataset into a BatchGenerator
train_datagen = DatasetGenerator(NpDataset(xtr, y=ytr), batch_size=64, shuffle=True, seed=1234)
# Turn the val data into a BatchGenerator
val_datagen = DatasetGenerator(NpDataset(xval, y=yval), batch_size=1000, shuffle=True, seed=1234)

# Fit the model
model.fit_generator(train_datagen, epochs=10,
                    steps_per_epoch=train_datagen.steps_per_epoch,
                    validation_data=val_datagen,
                    validation_steps=val_datagen.steps_per_epoch,
                    verbose=1)
