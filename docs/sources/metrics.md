### topk_accuracy


```python
topk_accuracy(output, target, topk)
```


Computes the precision@k for the specified values of k

__Arguments__

outputs -- A torch FloatTensor of arbitrary shape
targets -- A torch LongTensor of the same size except along
the channels dimension (the target dimension - 1)
topk -- The k to compute the topk accuracy for
- ____Returns__:__

A scalar tensor equal to the topk accuracy of the output


----

### accuracy


```python
accuracy(output, target)
```
