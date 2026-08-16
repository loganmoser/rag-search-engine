from .search_utils import DEFAULT_SEARCH_LIMIT,CACHE_DIR, load_movies, load_stop_words
import string
from nltk.stem import PorterStemmer
import pickle
import os
from collections import defaultdict, Counter
import sys
import math

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict(int, Counter) = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

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
        with open(self.term_freq_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        return math.log((len(self.docmap) +1) / (len(self.get_documents(term)) + 1))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def load(self):
        try:
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.docmap = pickle.load(f)
            with open(self.term_freq_path, 'rb') as f:
                self.term_frequencies = pickle.load(f)
        except Exception as e:
            print(f"Tried to read file. Error: {e}")


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f'Error in load function: {e}')
        sys.exit(1)
    results = []

    doc_ids = set()
    query_tokens = tokenize_text(query)
    for token in query_tokens:
        ids = idx.get_documents(token)
        for id in ids:
            if id in doc_ids:
                continue
            doc_ids.add(id)
            doc = idx.docmap[id]
            results.append(doc)
            if len(results) >= limit:
                return results
    return results

def tf_command(doc_id: int, term:str) -> None:
    idx = InvertedIndex()
    token_term = tokenize_word(term)
    
    try:
        idx.load()
    except Exception as e:
        print(f"Error loading  index and docmap: {e}")
        sys.exit(1)

    print(f"Term frequency for {term}: {idx.get_tf(doc_id, term)}")

def idf_command(term: str) -> None:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f'Error loading index and docmap: {e}')

    token_term = tokenize_word(term)

    idf = idx.get_idf(token_term)

    print(f"Iverse document frequency of {term}: {idf:.2f}")

def tfidf_command(doc_id: int, term: str) -> None:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f"Error loading index and docmap: {e}")
    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)

    tf_idf = tf * idf

    print(f"Inverse document frequency of {term}: {tf_idf:.2f}")

def preprocess_text(text:str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

STOP_WORDS = [preprocess_text(x) for x in load_stop_words()]

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = [stemmer.stem(token) for token in text.split() if token is not None and token not in STOP_WORDS]
    return tokens

def tokenize_word(text: str) -> str:
    text = tokenize_text(text)
    if len(text) != 1:
        raise Exception("Tokenizer didn't return just one word")
    return text[0]

