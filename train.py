import numpy as np
import pickle
from preprocessing import load_and_preprocess, generate_all_pairs
from negative_sampling import compute_sampling_probs, compute_negative_samples_batch
from model import initialize_matrices, forward_pass_batch, backward_pass_batch
import time


def compute_loss(pos_errors, neg_errors):
    """
    Computes the average loss for Skip-gram with negative sampling.

    Args:
        pos_errors (np.array): positive pair errors from forward pass, shape (N,)
        neg_errors (np.array): negative pair errors from forward pass, shape (N, k)

    Returns:
        float: average loss over the batch
    """

    pos_loss = -np.sum(np.log(1 + pos_errors + 1e-10))   
    neg_loss = -np.sum(np.log(1 - neg_errors + 1e-10))
    return (pos_loss + neg_loss) / len(pos_errors)


def train(
    data_path="data/text8",
    cache_file="data/cache.pkl",
    max_tokens=3_000_000,
    min_count = 5,
    embedding_dim=100,
    window_size=5,
    num_negative=7,
    lr_start=0.025,
    num_epochs=20,
    batch_size=512,
    save_path="data/embeddings.pkl",
    log_every=1000   
):
    """
    Trains a Skip-gram model with negative sampling on the text8 dataset.

    Args:
        data_path (str): path to text8 dataset
        cache_file (str): path for caching preprocessed data
        max_tokens (int): maximum number of tokens to load (~17M total)
        min_count (int): minimum frequency threshold for words
        embedding_dim (int): dimensionality of embeddings
        window_size (int): context window size on each side
        num_negative (int): number of negative samples per positive pair
        lr_start (float): initial learning rate
        num_epochs (int): number of training epochs
        batch_size (int): size of each training batch
        save_path (str): path to save final embeddings
        log_every (int): log average loss every N batches
    """
    
    # ------------------------------------------------------------------ #
    # 1. Loading and preprocessing
    # ------------------------------------------------------------------ #

    print("Loading and preprocessing data...")
    tokens, frequencies, word_to_idx, idx_to_word = load_and_preprocess(
        path=data_path,
        max_tokens=max_tokens,
        min_count = min_count,
        cache_file=cache_file
    )


    V = len(word_to_idx)
    print(f"Vocabulary size: {V}, Total tokens: {len(tokens)}")

    tokens = np.array([word_to_idx[word] for word in tokens], dtype=np.int32)

    # ------------------------------------------------------------------ #
    # 2. Initialization
    # ------------------------------------------------------------------ #

    W_in, W_out = initialize_matrices(V, embedding_dim)
    words, probs = compute_sampling_probs(frequencies, word_to_idx)
    
    # ------------------------------------------------------------------ #
    # 3. Generating all pairs
    # ------------------------------------------------------------------ #

    print("Generating all pairs...")
    center_indices, context_indices = generate_all_pairs(tokens, window_size)
    N = len(center_indices)
    print(f"Total pairs: {N}")

    # ------------------------------------------------------------------ #
    # 4. Training loop
    # ------------------------------------------------------------------ #

    loss_history = []
    total_steps = num_epochs * (N // batch_size)
    current_step = 0

    for epoch in range(num_epochs):

        perm = np.random.permutation(N)
        center_indices  = center_indices[perm]
        context_indices = context_indices[perm]

        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        
        window_loss = 0.0  
        batch_loss_history = []

        for i in range(0, N, batch_size):

            progress = current_step / total_steps
            lr = lr_start * (1 - progress)
            lr = max(lr, lr_start * 0.0001) 
            current_step += 1

            c_batch  = center_indices[i:i+batch_size]
            ctx_batch = context_indices[i:i+batch_size]

            neg_batch = compute_negative_samples_batch(
                c_batch, words, probs, k = num_negative
            )

            pos_errors, neg_errors = forward_pass_batch(
                c_batch, ctx_batch, neg_batch, W_in, W_out
            )


            backward_pass_batch(
                c_batch, ctx_batch, neg_batch,
                W_in, W_out, lr, pos_errors, neg_errors
            )

            window_loss += compute_loss(pos_errors, neg_errors)
            batches_done = (i // batch_size) + 1
            
            if batches_done % log_every == 0:
                avg_loss = window_loss / log_every
                print(f"Epoch {epoch+1} | Batch {batches_done} | Avg Loss: {avg_loss:.4f}")
                batch_loss_history.append(avg_loss)
                window_loss = 0.0

        if batch_loss_history:
            epoch_avg_loss = sum(batch_loss_history) / len(batch_loss_history)
            loss_history.append(epoch_avg_loss)
            print(f"Epoch {epoch+1} avg loss: {epoch_avg_loss:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Embedding savings
    # ------------------------------------------------------------------ #

    print(f"\nSaving embeddings to {save_path}...")
    with open(save_path, "wb") as f:
        pickle.dump({
            "W_in": W_in,
            "W_out": W_out,
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word
        }, f)
    print("Done!")

    return W_in, W_out, word_to_idx, idx_to_word, loss_history

if __name__ == "__main__":
    start = time.perf_counter()

    train()

    end = time.perf_counter()
    elapsed_minutes = (end - start) / 60
    print(f"Execution time: {elapsed_minutes:.2f} minutes")
