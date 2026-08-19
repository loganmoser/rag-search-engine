from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from .search_utils import(
    MOVIE_EMBEDDINGS_PATH,
    DATA_PATH,
)

class SemanticSearch:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text: str) -> list[str]:
        if len(text.strip()) == 0:
            raise ValueError("No text to embed.")

        embedding = self.model.encode([text])[0]

        return embedding

    def build_embeddings(self, documents: list[dict[int: str]]) -> list[str]:
        self.documents = documents
        for i, doc in enumerate(documents):
            self.document_map[i] = doc
        movie_strings = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings

    def load_or_build_embeddings(self, documents: list[dict[int: str]]):
        self.documents = documents
        for i, doc in enumerate(documents):
            self.document_map[i] = doc
        if os.path.isfile(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        else:
            return self.build_embeddings(documents)

    def search(self, query:str, limit: int):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_build_embeddings` first")
        query_embedding = self.generate_embedding(query)
        scores = []
        for embedding, i in zip(self.embeddings, self.document_map):
            cos_sim = cosine_similarity(query_embedding, embedding)
            scores.append((cos_sim, i)) # Use document map id to get title and description of movie

        sorted_scores = sorted(scores, key = lambda x: x[0], reverse=True)[:limit]

        results = [{"score": score, 
                    "title": self.document_map[i]['title'], 
                    "description": self.document_map[i]['description']} for score, i in sorted_scores]

        return results


def search_command(query: str, limit: int):
    semantic_instance = SemanticSearch()
    with open(DATA_PATH, 'r') as f:
        documents = json.load(f)['movies']
    semantic_instance.load_or_build_embeddings(documents)
    results = semantic_instance.search(query, limit)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} (score: {result['score']}){result['description'][:100]}...")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def verify_embeddings():
    semantic_instance = SemanticSearch()
    with open(DATA_PATH, 'r') as f:
        documents = json.load(f)['movies']
    semantic_instance.load_or_build_embeddings(documents)
    print(f"Numer of docs: {len(semantic_instance.documents)}")
    print(
        f"Embeddings shape: {semantic_instance.embeddings.shape[0]} vectors in {semantic_instance.embeddings.shape[1]} dimensions"
    )

def embed_query(query: str) -> None:
    semantic_instance = SemanticSearch()
    embedding = semantic_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def embed_text(text: str) -> None:
    semantic_instance = SemanticSearch()
    embedding = semantic_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model() -> None:
    semantic_instance = SemanticSearch()
    print(f"Model loaded: {semantic_instance.model}")
    print(f"Max sequence length: {semantic_instance.model.max_seq_length}")
