import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
from sklearn.decomposition import PCA

from core.converter import lab_to_hex

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'


def load_embeddings(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as csv:
        data = pandas.read_csv(csv)

    vectors, rgbs = [], []
    for _, (_, L, a, b) in data.iterrows():
        vectors.append([L, a, b])
        rgbs.append(lab_to_hex(L, a, b))

    return np.array(vectors), np.array(rgbs)


def visualize_pca(vectors, rgbs):
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(10, 8))
    for i, (x, y) in enumerate(vectors_2d):
        plt.scatter(x, y, color=rgbs[i], s=100, edgecolors='k')

    plt.title("Phase 1 Anchor Embedding · PCA Projection")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    vectors, rgbs = load_embeddings("color_words.csv")
    visualize_pca(vectors, rgbs)


if __name__ == "__main__":
    main()
