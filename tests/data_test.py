import pytest
import math
import numpy as np
import pyjet.data as data


def np_is_sorted(arr, unique=False):
    if unique:
        return np.all(arr[:-1] < arr[1:])
    return np.all(arr[:-1] <= arr[1:])


def test_np_dataset_basic():
    dataset_length = 5000
    num_features = 10
    batch_size = 32
    x = np.linspace(
        0, dataset_length, num=dataset_length * num_features,
        endpoint=False).reshape((dataset_length, -1))
    y = np.arange(dataset_length)
    dataset = data.NpDataset(x, y=y)
    # We gave it labels so it should default to generating with them
    assert dataset.output_labels
    # Check the length
    assert len(dataset) == dataset_length
    # Check the batch creation works properly
    batch_indicies = np.arange(batch_size)
    batch = dataset.create_batch(batch_indicies)
    expected_batch = x[batch_indicies], y[batch_indicies]
    assert np.all(batch[0] == expected_batch[0]) and np.all(
        batch[1] == expected_batch[1])
    # Check that we can generate properly without output labels
    dataset.toggle_labels()
    assert not dataset.output_labels
    batch = dataset.create_batch(batch_indicies)
    expected_batch = x[batch_indicies]
    assert np.all(batch == expected_batch)


def test_np_dataset_split():
    dataset_length = 5000
    num_features = 10
    split = 0.2
    x = np.linspace(
        0, dataset_length, num=dataset_length * num_features,
        endpoint=False).reshape((dataset_length, -1))
    y = np.arange(dataset_length)
    dataset = data.NpDataset(x, y=y)
    # Basic split
    train, val = dataset.validation_split(
        split=split, shuffle=False, seed=None)
    # Check they are the right length and don't share any elements
    assert len(train) == int(math.ceil(dataset_length * (1 - split)))
    assert len(val) == int(dataset_length * split)
    assert len(np.intersect1d(train.x.flatten(), val.x.flatten())) == 0
    assert len(np.intersect1d(train.y.flatten(), val.y.flatten())) == 0
    # Check we didn't shuffle by seeing if dataset is still sorted
    assert np_is_sorted(train.x)
    assert np_is_sorted(train.y)
    assert np_is_sorted(val.x)
    assert np_is_sorted(val.y)
    # Try splitting with shuffling
    seed = 1234
    train, val = dataset.validation_split(split=split, shuffle=True, seed=seed)
    # Check they are the right length and don't share any elements
    assert len(train) == int(math.ceil(dataset_length * (1 - split)))
    assert len(val) == int(dataset_length * split)
    assert len(np.intersect1d(train.x.flatten(), val.x.flatten())) == 0
    assert len(np.intersect1d(train.y.flatten(), val.y.flatten())) == 0
    # Check that the x and y match up
    assert np.all(train.y == train.x[:, 0])
    assert np.all(val.y == val.x[:, 0])
    # Check that it isn't sorted (Extremely unlikely this fails)
    assert not np_is_sorted(train.x)
    assert not np_is_sorted(train.y)
    assert not np_is_sorted(val.x)
    assert not np_is_sorted(val.y)
    # Check that splitting with same seed produces the same result
    train2, val2 = dataset.validation_split(
        split=split, shuffle=True, seed=seed)
    assert np.all(train2.x == train.x)
    assert np.all(train2.y == train.y)
    assert np.all(val2.x == val.x)
    assert np.all(val2.y == val.y)


def test_dataset_generator_creation():
    # Create a dataset to generate
    dataset_length = 5000
    num_features = 10
    batch_size = 32
    steps_per_epoch = int(math.ceil(dataset_length / batch_size))
    x = np.linspace(
        0, dataset_length, num=dataset_length * num_features,
        endpoint=False).reshape((dataset_length, -1))
    y = np.arange(dataset_length)
    # TODO: This is bad practice and should be removed into a
    # seperate testable function
    # Test that inferring the steps_per_epoch and batch_size works properly
    dataset = data.NpDataset(x, y=y)
    gen = data.DatasetGenerator(
        dataset,
        steps_per_epoch=None,
        batch_size=batch_size,
        shuffle=False,
        seed=None)
    assert gen.steps_per_epoch == steps_per_epoch
    assert gen.batch_size == batch_size
    gen = data.DatasetGenerator(
        dataset,
        steps_per_epoch=steps_per_epoch,
        batch_size=None,
        shuffle=False,
        seed=None)
    assert gen.steps_per_epoch == steps_per_epoch
    assert gen.batch_size == batch_size


def test_dataset_generator_basic():
    dataset_length = 50
    num_features = 10
    batch_size = 32
    x = np.linspace(
        0, dataset_length, num=dataset_length * num_features,
        endpoint=False).reshape((dataset_length, -1))
    y = np.arange(dataset_length)
    dataset = data.NpDataset(x, y=y)
    gen = data.DatasetGenerator(
        dataset,
        steps_per_epoch=None,
        batch_size=batch_size,
        shuffle=False,
        seed=None)
    # Test that batches are right without shuffling
    batch_arguments = np.arange(dataset_length)
    for i in range(gen.steps_per_epoch):
        x_batch, y_batch = next(gen)
        assert np.all(x_batch == x[batch_arguments[i * batch_size:(
            i + 1) * batch_size]])
        assert np.all(y_batch == y[batch_arguments[i * batch_size:(
            i + 1) * batch_size]])
    # Test that with the seed and shuffling, the generation is correct
    seed = 1234
    np.random.seed(seed)
    np.random.shuffle(batch_arguments)
    gen = data.DatasetGenerator(
        dataset,
        steps_per_epoch=None,
        batch_size=batch_size,
        shuffle=True,
        seed=seed)
    for i in range(gen.steps_per_epoch):
        x_batch, y_batch = next(gen)
        assert np.all(x_batch == x[batch_arguments[i * batch_size:(
            i + 1) * batch_size]])
        assert np.all(y_batch == y[batch_arguments[i * batch_size:(
            i + 1) * batch_size]])

    for i in range(gen.steps_per_epoch):
        x_batch, y_batch = next(gen)
        assert np.all(
            x_batch != x[batch_arguments[i * batch_size:(i + 1) * batch_size]])
        assert np.all(
            y_batch != y[batch_arguments[i * batch_size:(i + 1) * batch_size]])
