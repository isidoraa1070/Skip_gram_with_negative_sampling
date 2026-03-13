import numpy as np

def compute_sampling_probs(frequencies, word_to_idx):
    """
    Computes negative sampling probabilities using f(w)^0.75 (Mikolov et al., 2013).

    Returns word indices and their normalized probabilities for negative sampling.
    """
    words = np.array([word_to_idx[w] for w in frequencies.keys()], dtype=np.int32)
    freqs = np.array(list(frequencies.values()), dtype=np.float64)
    
    modified = freqs ** 0.75
    probs = modified / modified.sum()
    
    return words, probs


def compute_negative_samples_batch(center_indices, words, probs, k=3):
    """
    Samples k negative words for each center-context pair in a batch.

    Returns a (N, k) array of negative sample indices.
    """
    N = len(center_indices)
    neg_samples = np.random.choice(words, size=(N, k), p=probs)
    return neg_samples
    







    

