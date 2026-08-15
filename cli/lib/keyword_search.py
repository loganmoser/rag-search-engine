from .search_utils import DEFAULT_SEARCH_LIMIT,CACHE_DIR, load_movies, load_stop_words
import string
from nltk.stem import PorterStemmer
import pickle
import os

stemmer = PorterStemmer()

class InvertedIndex():
    
    def __init__(self):
        self.index: dict[str: set()] = dict()
        self.docmap: dict[int: str] = dict()

    def __add_document(self, doc_id:int, text:str) -> None:
        tokens = tokenize(text)

        for token in tokens:
            if token not in self.index.keys():
                self.index[token] = {doc_id}
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
            os.mkdir(CACHE_DIR)
        # pickle the index and docmap
        index_file_path = os.path.join(CACHE_DIR, "index.pkl")
        docmap_file_path = os.path.join(CACHE_DIR, "docmap.pkl")
        with open(index_file_path, 'wb') as f:
            pickle.dump(self.index, f)
        with open(docmap_file_path, 'wb') as f:
            pickle.dump(self.docmap, f)

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        query_tokens = tokenize(query)
        title_tokens = tokenize(movie["title"])
        for query_token in query_tokens:
            for title_token in title_tokens:
                if query_token in title_token:
                    results.append(movie)
            if len(results) >= limit:
                break
    return results


def preprocess_text(text:str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

STOP_WORDS = [preprocess_text(x) for x in load_stop_words()]

def tokenize(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = [stemmer.stem(token) for token in text.split() if token is not None and token not in STOP_WORDS]
    return tokens
