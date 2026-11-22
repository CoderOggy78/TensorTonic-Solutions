import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    result = np.full((len(seqs), max_len), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        result[i, :length] = seq[:length]

    return result