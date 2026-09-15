from ast import Break

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re
from .search_utils import(
    Movie,
    MOVIE_EMBEDDINGS_PATH,
    DATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    load_movies
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

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for i, doc in enumerate(self.documents):
            self.document_map[i] = doc
        all_chunks: list[str] = []
        metadata: list[dict] = []

        for i, doc in enumerate(self.documents):
            desc = doc.get('description')
            if not desc:
                continue # If document description is empty, skip it
            doc_chunks = semantic_chunk(desc, 4, 1)
            for j, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata = {"movie_idx": i,
                    "chunk_idx": j,
                    "total_chunks": len(doc_chunks)}
                metadata.append(chunk_metadata)
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for i, doc in enumerate(self.documents):
            self.document_map[i] = doc
        if os.path.isfile(CHUNK_EMBEDDINGS_PATH) and os.path.isfile(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, 'r') as f:
                self.chunk_metadata = json.load(f)
            if len(self.chunk_embeddings) == len(documents):
                return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

def embed_chunks():
    chunked_search = ChunkedSemanticSearch()
    documents = load_movies()
    chunked_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(chunked_search.chunk_embeddings)} chunked embeddings")


def search_command(query: str, limit: int):
    semantic_instance = SemanticSearch()
    with open(DATA_PATH, 'r') as f:
        documents = json.load(f)['movies']
    semantic_instance.load_or_build_embeddings(documents)
    results = semantic_instance.search(query, limit)
    for i, result in enumerate(results):
        print(f"{i+1}. {result['title']} (score: {result['score']}){result['description'][:100]}...")

def semantic_chunk(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = 0):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    for i in range(0, len(sentences), max_chunk_size-overlap):
        chunked_sentence = sentences[i:+i+max_chunk_size]
        chunks.append(" ".join(chunked_sentence))
        if i+max_chunk_size >= len(sentences):
            break
    return chunks


def chunk_command(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    split_text = text.split(" ")
    chunked_strings = []
    for i in range(0, len(split_text), chunk_size-overlap):
        chunked_strings.append(" ".join(split_text[i:i+chunk_size]))
    return chunked_strings

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
