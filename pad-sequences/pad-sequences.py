import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N = len(seqs)

    if max_len is None:
        L = max(map(len, seqs), default=0)
    else:
        L = max_len

    if N:
        dtype = np.result_type(
            pad_value,
            *[np.asarray(seq).dtype for seq in seqs]
        )
    else:
        dtype = np.asarray(pad_value).dtype

    result = np.full((N, L), pad_value, dtype=dtype)

    for i, seq in enumerate(seqs):
        n = min(len(seq), L)
        result[i, :n] = seq[:n]

    return result