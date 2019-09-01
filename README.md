# PyJet (WIP)

[![Build Status](https://travis-ci.org/abhmul/PyJet.svg?branch=master)](https://travis-ci.org/abhmul/PyJet)
[![Coverage Status](https://coveralls.io/repos/github/abhmul/PyJet/badge.svg?branch=master)](https://coveralls.io/github/abhmul/PyJet?branch=master)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/abhmul/PyJet/blob/master/LICENSE)

Custom Pytorch Frontend I gradually build on for my projects. This has pretty much become a multi-year personal project I use to keep a personally customized front-end for deep learning when trying out new projects. Over the years I've added all kinds of cool features like:
* Input size inferral - no need to define the input size to your layers
* Interface to train models with multiple or arbitrarily complex optimizer schemes
* Interface to train models with multiple or arbitrarily complex loss schemes
* Tracking training stats that can persist state over multiple batches
* Keras-like training and usage of models
* Pythonic pipelining and management of data
* Integration with useful libraries like `imgaug` and `tqdm`

Below is the console and graph output from running [`mnist.py`](https://github.com/abhmul/PyJet/blob/master/examples/mnist.py) in the *examples* folder
![Example Console](https://github.com/abhmul/PyJet/raw/master/examples/example_console.png)

![Example Console](https://github.com/abhmul/PyJet/raw/master/examples/example_graph.png)


## Installation

To install run the following in your HOME directory.

```
git clone https://github.com/abhmul/PyJet/
cd PyJet
sudo pip install -e .
cd ..
```

## Update

To update, go to your PyJet installation directory (should be HOME/PyJet if you followed the installation instructions) and run

```
git pull
```
