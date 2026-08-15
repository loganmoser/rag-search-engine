import json
import os
from typing import Any, TypedDict
import pickle

class Movie(TypedDict):
    id: int
    title: str 
    description: str

class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]

class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]

class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]

class InvertedIndex():
    index: dict[str: set()]
    docmap: dict[int: str]

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = tokenize(text)

        for token in tokens:
            if token not in self.index.keys():
                self.index[token] = set(doc_id)
            else:
                self.index[token].add(doc_id)
    
    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index[term])

    def build(self) -> None:
        movies = load_movies()

        for movie in movies:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
    
    def save(self) -> None:
        if not os.path.isdir(CACHE_DIR):
            os.path.mkdir(CACHE_DIR)

        # pickle the index and docmap
        index_file_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_file_path = os.join(CACHE_DIR, "docmap.pkl")
        pickle.dump(self.index, index_file_path)
        pickle.dump(self.docmap, docmap_file_path)


DEFAULT_ALPHA = 0.5
RRF_K = 60
SEARCH_MULTIPLIER = 5

DEFAULT_SEARCH_LIMIT = 5
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        words = f.read()
        words_list = words.splitlines()

        return words_list


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def load_golden_dataset() -> GoldenDataset:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)

