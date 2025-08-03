import json


def load_color_embeddings(path):
    with open(path, 'r', encoding='utf-8') as f:
        embedding = json.load(f)
    return embedding
