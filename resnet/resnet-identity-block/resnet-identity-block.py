import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """

    W1 = np.array(W1)
    W2 = np.array(W2)
    x = np.array(x)
    
    x_skip = x.copy()

    x = np.maximum(0, x@W1.T)
    x = np.maximum(0, x@W2.T + x_skip)

    return x
    
    