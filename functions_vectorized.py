import numpy as np


def prod_non_zero_diag(x: np.ndarray):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    return (x.diagonal() + (x.diagonal() == 0)).prod()


def are_multisets_equal(x: np.ndarray, y: np.ndarray):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """
    return np.equal(np.sort(x), np.sort(y))


def max_after_zero(x: np.ndarray):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """
    y = np.where(x == 0, 1, 0)
    y[0] = 0
    y = np.roll(y, 1)
    return (x * y).max()


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """
    return np.sum(img * coefs, axis=2)


def run_length_encoding(x: np.ndarray):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    n = len(x)
    y = np.diff(x)
    z = np.append(np.insert(np.where(y != 0), 0, 0), n - 1)
    F = np.diff(z)
    F[0] += 1
    return x[np.cumsum(np.diff(z))], F


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    z = np.sum((x[:,np.newaxis] - y[np.newaxis,:])**2, axis=2)
    return np.sqrt(z)
