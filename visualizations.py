import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import csv
from evaluate import word_pair_similarity, load_embeddings


def plot_tsne(W_in, word_to_idx, words_to_plot):
    """
    Visualizes selected word embeddings in 2D using t-SNE.
    
    Shows:
       similar words are close in space
       clusters (e.g., countries together, music terms together)

    Args:
        W_in (np.array): Input embedding matrix, shape (V, d)
        word_to_idx (dict): Mapping from word to index
        words_to_plot (list of str): Words to visualize
        save_path (str): Path to save the plot
    """
    vectors = []
    labels = []
    for word in words_to_plot:
        if word in word_to_idx:
            vectors.append(W_in[word_to_idx[word]])
            labels.append(word)
    
    vectors = np.array(vectors)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    vectors_2d = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
    
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]))
    
    plt.title("t-SNE Word Embeddings")
    plt.savefig("visualizations/tsne.png")



def plot_similarity_heatmap(W_in, word_to_idx, words):
    """
    Plots a similarity heatmap between selected words using their embeddings.

    Shows:
       how words relate to each other
       clearly visible clusters (e.g., gender, countries, animals)

    Args:
        W_in (np.array): Input embedding matrix, shape (V, d)
        word_to_idx (dict): Mapping from word to index
        words (list of str): Words to include in heatmap
        save_path (str): Path to save the heatmap image
    """
    vectors = np.array([W_in[word_to_idx[w]] 
                       for w in words 
                       if w in word_to_idx])
    

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)
    
    sim_matrix = np.dot(vectors_norm, vectors_norm.T)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(words)), words, rotation=45)
    plt.yticks(range(len(words)), words)
    plt.title("Word Similarity Heatmap")
    plt.tight_layout()
    plt.savefig("visualizations/similarity_heatmap.png")


def plot_wordsim_scatter(human_scores, model_scores, correlation):
    """
    Shows:
      how well the model agrees with human similarity judgments
      highlights potential outliers where the model disagrees
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(human_scores, model_scores, alpha=0.5)
    plt.xlabel("Human Scores")
    plt.ylabel("Model Scores")
    plt.title(f"WordSim-353 | Spearman: {correlation:.4f}")
    plt.savefig("visualizations/wordsim_scatter.png")

def get_wordsim_scores(W_in, word_to_idx, wordsim_path="data/wordsim353crowd.csv"):
    """
    Returns human scores, model scores, and Spearman correlation for scatter plot.

    Args:
        W_in (np.array): input embeddings matrix
        word_to_idx (dict): mapping from word to index
        wordsim_path (str): path to WordSim-353 CSV file

    Returns:
        tuple: (human_scores, model_scores, correlation)
    """
    from scipy.stats import spearmanr

    model_scores = []
    human_scores = []

    try:
        with open(wordsim_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                w1, w2, score = row[0], row[1], row[2]
                if w1 in word_to_idx and w2 in word_to_idx:
                    sim = word_pair_similarity(w1, w2, W_in, word_to_idx)
                    model_scores.append(sim)
                    human_scores.append(score)

        correlation, _ = spearmanr(model_scores, human_scores)
        return human_scores, model_scores, correlation

    except FileNotFoundError:
        print(f"WordSim-353 file not found. Skipping scatter plot.")
        return None, None, None


def visualize(embeddings_path="data/embeddings.pkl", loss_history=None):
    """
    Run all visualizations for embeddings.

    1. Loss curve over training
    2. t-SNE projection of selected words
    3. Word similarity heatmap
    4. WordSim-353 scatter plot

    Args:
        embeddings_path (str): path to saved embeddings (.pkl)
        loss_history (list or None): optional list of training losses
    """
    
    W_in, word_to_idx, _ = load_embeddings(embeddings_path)

    # --- 1. t-SNE ---
    tsne_words = [
        # Royalty
        "king", "queen", "prince", "princess",
        # Geography
        "paris", "berlin", "london", "rome",
        "france", "germany", "england", "italy","seattle", "mexico",
        # Technology
        "computer", "laptop", "desktop", "software",
        # Music
        "music", "dance", "opera", "jazz","punk", "hip hop",
        # Politics
        "communist", "anarchist", "democracy", "fascist","marxism", "revolution", "libertarian",
        # Autors
        "bakunin", "kropotkin", "goldman", "warren", "godwin", "tolstoy"
    ]
    plot_tsne(W_in, word_to_idx, tsne_words)
    print("t-SNE saved to tsne.png")

    # --- 2. Similarity Heatmap ---
    heatmap_words = [
        "king", "queen", "man", "woman",
        "paris", "berlin", "france", "germany",
        "computer", "laptop", "music", "dance",
        "marxism", "communist", "property", "wealth"
    ]
    plot_similarity_heatmap(W_in, word_to_idx, heatmap_words)
    print("Heatmap saved to similarity_heatmap.png")

    # --- 3. WordSim Scatter ---
    human_scores, model_scores, correlation = get_wordsim_scores(W_in, word_to_idx)
    if human_scores is not None:
        plot_wordsim_scatter(human_scores, model_scores, correlation)
        print("WordSim scatter saved to wordsim_scatter.png")

if __name__ == "__main__":
    visualize()




