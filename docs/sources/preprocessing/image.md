<span style="float:right;">[[source]](https://github.com/PyJet/tree/master/pyjet/pyjet/preprocessing/image.py#L84)</span>
### ImageDataGenerator

```python
pyjet.preprocessing.image.ImageDataGenerator(generator, labels=True, augment_masks=True, samplewise_center=False, samplewise_std_normalization=False, rotation_range=0.0, width_shift_range=0.0, height_shift_range=0.0, shear_range=0.0, zoom_range=0.0, channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False, vertical_flip=False, rescale=None, preprocessing_function=None, seed=None, data_format='channels_last')
```

Generate minibatches of image data with real-time data augmentation.
__Arguments__

- __samplewise_center__: set each sample mean to 0.
- __samplewise_std_normalization__: divide each input by its std.
- __rotation_range__: degrees (0 to 180).
- __width_shift_range__: fraction of total width.
- __height_shift_range__: fraction of total height.
- __shear_range__: shear intensity (shear angle in radians).
- __zoom_range__: amount of zoom. if scalar z, zoom will be randomly picked
in the range [1-z, 1+z]. A sequence of two can be passed instead
to select this range.
- __channel_shift_range__: shift range for each channels.
- __fill_mode__: points outside the boundaries are filled according to the
given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
is 'nearest'.
- __cval__: value used for points outside the boundaries when fill_mode is
'constant'. Default is 0.
- __horizontal_flip__: whether to randomly flip images horizontally.
- __vertical_flip__: whether to randomly flip images vertically.
- __rescale__: rescaling factor. If None or 0, no rescaling is applied,
otherwise we multiply the data by the value provided. This is
applied after the `preprocessing_function` (if any provided)
but before any other transformation.
- __preprocessing_function__: function that will be implied on each input.
The function will run before any other modification on it.
The function should take one argument:
one image (Numpy tensor with rank 3),
and should output a Numpy tensor with the same shape.
- __data_format__: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
(the depth) is at index 1, in 'channels_last' mode it is at index 3.
It defaults to "channels_last".

----

### standardize


```python
standardize(self, x)
```


Apply the normalization configuration to a batch of inputs.
__Arguments__

- __x__: batch of inputs to be normalized.
__Returns__

The inputs, normalized.

----

### random_transform


```python
random_transform(self, x, seed=None)
```


Randomly augment a single image tensor.
__Arguments__

- __x__: 3D tensor, single image.
- __seed__: random seed.
__Returns__

A randomly transformed version of the input (same shape).
