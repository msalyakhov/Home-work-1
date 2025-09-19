import numpy as np
import math

def prod_non_zero_diag(x: list):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """
    prod = 1
    n, m = len(x), len(x[0])
    for i in range(min(n, m)):
        if x[i][i] != 0:
            prod *= x[i][i]
    return prod


def are_multisets_equal(x: list, y: list):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """
    x.sort()
    y.sort()
    return x == y


def max_after_zero(x: list):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """
    res = 0
    for i in range(1, len(x)):
        if x[i - 1] == 0:
            res = max(res, x[i])
    return res


def convert_image(img: np.ndarray, coefs: np.ndarray):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """
    n = len(img)
    m = len(img[0])
    k = len(img[0][0])
    res = [[0 for i in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            for k in range(k):
                res[i][j] += coefs[k] * img[i][j][k]
    return res


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """
    n = len(x)
    a = [x[0]]
    b = [1]
    cnt = 1
    for i in range(1, n):
        if x[i] != x[i - 1]:
            a.append(x[i])
            b.append(0)
        b[-1] += 1
    return a, b
    


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """
    n = len(x)
    m = len(y)
    ans = [[0.0 for i in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            ans[i][j] = math.sqrt((x[i][0] - y[j][0]) ** 2 + (x[i][1] - y[i][1]) ** 2)
    return ans
