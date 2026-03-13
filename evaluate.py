import numpy as np
import pickle
import csv

def load_embeddings(path="data/embeddings.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["W_in"], data["word_to_idx"], data["idx_to_word"]


def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Returns:
        float: similarity score in range [-1, 1]
    """
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / (norm + 1e-10)


def most_similar(word, W_in, word_to_idx, idx_to_word, top_n=5):
    """
    Finds the top-N most similar words to a given word based on cosine similarity.

    Args:
        word (str): target word
        W_in (np.array): input embedding matrix, shape (V, d)
        word_to_idx (dict): mapping from word to index
        idx_to_word (dict): mapping from index to word
        top_n (int): number of most similar words to return

    Returns:
        list of tuples: (word, similarity_score)
    """
    if word not in word_to_idx:
        print(f"Word '{word}' not in vocabulary.")
        return []

    idx = word_to_idx[word]
    vec = W_in[idx]

    
    similarities = np.dot(W_in, vec) 
    norms = np.linalg.norm(W_in, axis=1) * np.linalg.norm(vec)
    similarities = similarities / (norms + 1e-10) 


    top_indices = np.argpartition(similarities, -top_n)[-top_n:]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

    results = [(idx_to_word[i], float(similarities[i])) for i in top_indices]
    return results


def word_analogy(word_a, word_b, word_c, W_in, word_to_idx, idx_to_word, top_n=5):
    """
    Solves word analogies: word_a - word_b + word_c ≈ ?

    Args:
        word_a, word_b, word_c (str): words forming the analogy
        W_in (np.array): input embedding matrix, shape (V, d)
        word_to_idx (dict): mapping from word to index
        idx_to_word (dict): mapping from index to word
        top_n (int): number of top predictions to return

    Returns:
        list of tuples: (word, similarity_score)
    """
    for w in [word_a, word_b, word_c]:
        if w not in word_to_idx:
            print(f"Word '{w}' not in vocabulary.")
            return []

    vec_a = W_in[word_to_idx[word_a]]
    vec_b = W_in[word_to_idx[word_b]]
    vec_c = W_in[word_to_idx[word_c]]

    target_vec = vec_a - vec_b + vec_c
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-10)


    norms = np.linalg.norm(W_in, axis=1, keepdims=True)
    W_normalized = W_in / (norms + 1e-10)
    similarities = np.dot(W_normalized,target_vec)

    exclude = {word_to_idx[word_a], word_to_idx[word_b], word_to_idx[word_c]}
    top_indices = [
        i for i in np.argsort(similarities)[::-1]
        if i not in exclude
    ][:top_n]

    results = [(idx_to_word[i], float(similarities[i])) for i in top_indices]
    return results


def word_pair_similarity(word1, word2, W_in, word_to_idx):
    """
    Computes cosine similarity between two words based on the embeddings.

    Args:
        word1, word2 (str): words to compare
        W_in (np.array): input embedding matrix, shape (V, d)
        word_to_idx (dict): mapping from word to index

    Returns:
        float or None: similarity score in [-1, 1], or None if a word is out of vocabulary
    """
    if word1 not in word_to_idx or word2 not in word_to_idx:
        print("One or both words not in vocabulary.")
        return None

    vec1 = W_in[word_to_idx[word1]]
    vec2 = W_in[word_to_idx[word2]]
    return cosine_similarity(vec1, vec2)


def evaluate_wordsim353(W_in, word_to_idx, wordsim_path="data/wordsim353crowd.csv"):
    """
    Evaluates embeddings on the WordSim-353 benchmark.

    Computes Spearman correlation between model-computed similarities
    and human-assigned similarity scores.

    Args:
        W_in (np.array): input embedding matrix, shape (V, d)
        word_to_idx (dict): mapping from word to index
        wordsim_path (str): path to WordSim-353 CSV file

    Returns:
        float or None: Spearman correlation, or None if file not found
    """
    from scipy.stats import spearmanr

    model_scores = []
    human_scores = []
    skipped = 0

    try:
        with open(wordsim_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # preskoči header red (Word1, Word2, HumanScore)
            for row in reader:
                w1, w2, score = row[0].lower(), row[1].lower(), float(row[2])

                if w1 in word_to_idx and w2 in word_to_idx:
                    sim = word_pair_similarity(w1, w2, W_in, word_to_idx)
                    model_scores.append(sim)
                    human_scores.append(score)
                else:
                    skipped += 1

        correlation, p_value = spearmanr(model_scores, human_scores)
        print(f"WordSim-353 | Spearman: {correlation:.4f} | p-value: {p_value:.16f} | Skipped: {skipped}")
        return correlation

    except FileNotFoundError:
        print(f"WordSim-353 file not found at {wordsim_path}. Skipping benchmark.")
        return None


def evaluate(embeddings_path="data/embeddings.pkl"):
    """
    Runs standard evaluation on trained embeddings.

    Evaluations include:
        - Top-N most similar words
        - Word analogy tasks
        - Optional WordSim-353 benchmark

    Args:
        embeddings_path (str): path to saved embeddings pickle file
    """
    W_in, word_to_idx, idx_to_word = load_embeddings(embeddings_path)

    # --- Most similar ---
    test_words = ["king", "computer", "france", "music", "berlin", "anarchists", "violent","massacres",
                  "bombings","communist","tolstoy","evolution", "conquest","terrorism"]
    for word in test_words:
        print(f"Most similar to '{word}':")
        results = most_similar(word, W_in, word_to_idx, idx_to_word, top_n=10)
        for w, score in results:
            print(f"  {w:<20} {score:.4f}")
        print()

    # --- Analogies ---
    analogies = [
        ("king", "man", "woman"),       
        ("paris", "france", "germany"),
        ("prince","man","woman"),
        ("anarchism", "anarchist", "communist")
    ]
    for a, b, c in analogies:
        print(f"'{a}' - '{b}' + '{c}' = ?")
        results = word_analogy(a, b, c, W_in, word_to_idx, idx_to_word)
        for w, score in results:
            print(f"  {w:<20} {score:.4f}")
        print()

    # --- WordSim-353 benchmark ---
    evaluate_wordsim353(W_in, word_to_idx)


if __name__ == "__main__":
    evaluate()