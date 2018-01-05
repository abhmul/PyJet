### bce_with_logits


```python
bce_with_logits(outputs, targets, size_average=True)
```



Computes the binary cross entropy between targets and output's logits.

See :class:`~torch.nn.BCEWithLogitsLoss` for details.

__Arguments__

outputs -- A torch FloatTensor of arbitrary shape with a 1 dimensional channel axis
targets -- A binary torch LongTensor of the same size without the channel axis
size_average -- By default, the losses are averaged over observations for each minibatch.
However, if the field size_average is set to False, the losses are instead
summed for each minibatch.
- ____Returns__:__

A scalar tensor equal to the total loss of the output.

- __Examples:__:

>>> input = autograd.Variable(torch.randn(3), requires_grad=True)
>>> target = autograd.Variable(torch.FloatTensor(3).random_(2))
>>> loss = bce_with_logits(input, target)
>>> loss.backward()

----

### categorical_crossentropy


```python
categorical_crossentropy(outputs, targets, size_average=True)
```



Computes the categorical crossentropy loss over some outputs and targets according the
equation for the ith output

-log(output[target])

and is accumulated with a sum or average over all outputs.

- ____Arguments__:__

outputs -- The torch FloatTensor output from a model with the shape (N, C) where N is the
number of outputs and C is the number of classes.
targets -- The torch LongTensor indicies of the ground truth with the shape (N,) where N is
the number of outputs and each target t is 0 <= t < C.
size_average -- By default, the losses are averaged over observations for each minibatch.
However, if the field size_average is set to False, the losses are instead
summed for each minibatch.
- ____Returns__:__

A scalar tensor equal to the total loss of the output.
