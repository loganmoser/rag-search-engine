from PIL import Image
from sentence_transformers import SentenceTransformer
import pathlib
import numpy as np
from .search_utils import Movie, load_movies

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents: list[Movie] = None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts)

    def search_with_image(self, image_path):
        image_embedding = self.embed_image(image_path)
        results = []
        for text_embedding, doc in zip(self.text_embeddings, self.documents):
            similarity_score = cosine_similarity(image_embedding, text_embedding)
            results.append(
                {
                    "id": doc['id'],
                    "title": doc['title'],
                    "description": doc['description'],
                    "similarity_score": similarity_score
                }
            )
        sorted_results = sorted(results, key = lambda x: x['similarity_score'], reverse=True)[:5]
        return sorted_results

    def embed_image(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return self.model.encode(img)


def verify_image_embedding(image_path: pathlib.Path) -> None:
    model = MultimodalSearch()
    embedding = model.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_serach_command(image_path):
    movies = load_movies()
    model = MultimodalSearch(documents = movies)
    return model.search_with_image(image_path)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
