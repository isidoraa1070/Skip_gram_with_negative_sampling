from collections import Counter
import os
import pickle
import numpy as np
import random

def load_and_preprocess(path="data/text8", max_tokens=3_000_000, min_count=5, cache_file="data/cache.pkl"):
    """
    Loads and preprocesses the text8 dataset for Skip-gram training.

    Only the first `max_tokens` words are loaded (~17M tokens total) 
    and very rare words (fewer than `min_count` occurrences) are removed.
    Frequent words are subsampled and the processed dataset is cached 
    for faster reuse.

    Additional format checks are implemented in utils/checkings.py.
    """

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print("Loading cached data...")
            return pickle.load(f)

    print("Processing dataset...")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = text.split()[:max_tokens]

    frequencies = Counter(tokens)
    tokens = [token for token in tokens if frequencies[token] >= min_count]

    print(f"Tokens before subsampling: {len(tokens)}")
    frequencies = Counter(tokens)
    tokens = subsampling_of_freq_words(tokens, frequencies, t = 1e-3)
    print(f"Tokens after subsampling: {len(tokens)}")

    frequencies = Counter(tokens)
    vocabulary = sorted(frequencies.keys())
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    
    result = (tokens, frequencies, word_to_idx, idx_to_word)

    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    print("Cached dataset created.")

    return result

def subsampling_of_freq_words(tokens, frequencies , t = 1e-3):
    
    """
    Subsamples frequent words using the probability:
    P(discard) = 1 - sqrt(t / f), where f is the word frequency, and t is a chosen threshold.

    Proposed in Mikolov et al. (2013) Word2Vec paper.
    """
    
    total_tokens = len(tokens)

    discard_prob = {
        w: max(0, 1 - np.sqrt(t / (frequencies[w] / total_tokens)))
        for w in frequencies
    }

    filtered_tokens = [
        w for w in tokens
        if random.random() > discard_prob[w]
    ]

    return filtered_tokens
           

def generate_all_pairs(tokens, window_size=5):

    """
    Generates all center-context pairs for Skip-gram training.

    Sliding a window of size `window_size` around each center word,
    all (center, context) index pairs are created.

    Returns:
        tuple: (center_indices, context_indices) as np.int32 arrays
    """
    tokens = np.array(tokens)
    center_indices = []
    context_indices = []

    for offset in range(-window_size, window_size + 1):
        if offset == 0:
            continue
        
        if offset > 0:
            center  = tokens[:-offset]
            context = tokens[offset:]
        else:
            center  = tokens[-offset:]
            context = tokens[:offset]
        
        center_indices.append(center)
        context_indices.append(context)
    
    center_indices  = np.concatenate(center_indices).astype(np.int32)   
    context_indices = np.concatenate(context_indices).astype(np.int32)  
    
    return center_indices, context_indices
