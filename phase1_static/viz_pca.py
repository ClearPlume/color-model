import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from phase1_static.data_loader import load_color_embeddings

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'


def load_embeddings(json_path):
    data = load_color_embeddings(json_path)
    tokens, vectors, rgbs = [], [], []
    for token, entry in data.items():
        emb = entry.get("embedding", None)
        if emb:
            tokens.append(token)
            vectors.append(emb)
            rgbs.append([v / 255 for v in entry["rgb"]])
    return tokens, np.array(vectors), np.array(rgbs)


def visualize_pca(tokens, vectors, rgbs):
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    for i, token in enumerate(tokens):
        print(f"{token}: PCA1 = {vectors_2d[i][0]:.4f}, PCA2 = {vectors_2d[i][1]:.4f}")

    plt.figure(figsize=(10, 7))
    for i, (x, y) in enumerate(vectors_2d):
        plt.scatter(x, y, color=rgbs[i], s=100, edgecolors='k')
        plt.text(x + 0.01, y, tokens[i], fontsize=10, ha='left', va='center')

    plt.title("Phase 1 Anchor Embedding · PCA Projection")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    json_path = "color_words.json"
    tokens, vectors, rgbs = load_embeddings(json_path)
    visualize_pca(tokens, vectors, rgbs)
