import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """

    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)
    x = np.array(x)

     
    shortcut= x @ Ws

    x = np.maximum(0, x @ W1)

    x = np.maximum(0, x @ W2 + shortcut)

    return x