from .search_utils import DEFAULT_SEARCH_LIMIT,CACHE_DIR, load_movies, load_stop_words
import string
from nltk.stem import PorterStemmer
import pickle
import os
from collections import defaultdict

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

def search_command(query: str, inverted_index: InvertedIndex, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    
    try:
        index, docmap = inverted_index.load()
        print(docmap)
    except Exception as e:
        print(f'Error in load function: {e}')
        exit
    results = []

    query_tokens = tokenize(query)
    for token in query_tokens:
        if token in index:
            ids = index[token]
            for id in ids:
                results.append(docmap[id])
                if len(results) >= limit:
                    break
    return results


def preprocess_text(text:str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

STOP_WORDS = [preprocess_text(x) for x in load_stop_words()]

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = [stemmer.stem(token) for token in text.split() if token is not None and token not in STOP_WORDS]
    return tokens
