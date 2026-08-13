import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output
    (compress, process, expand + skip)
    """

    W1 = np.array(W1)
    W2 = np.array(W2)
    W3 = np.array(W3)
    Ws = np.array(Ws)
    x = np.array(x)

    skip = x @ Ws

    x = np.maximum(0, x @ W1)  # compress
    x = np.maximum(0, x @ W2)  # process
    x = x @ W3                 # expand

    return np.maximum(0, x + skip)
