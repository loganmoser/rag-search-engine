from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    CACHE_DIR,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stop_words,
    Movie,
    SearchResult,
    format_search_result,
)
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
        self.term_frequencies: dict(int, Counter) = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_freq_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_bm25_tf(self, doc_id: int, term: str, k1:float=BM25_K1, b:float=BM25_B) -> int:
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        raw_tf = self.get_tf(doc_id, term)
        bm25_tf = (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)
        return bm25_tf

    def get_idf(self, term: str) -> float:
        return math.log((len(self.docmap) +1) / (len(self.get_documents(term)) + 1))

    def get_bm25_idf(self, term:str) -> float:
        return math.log((len(self.docmap) - len(self.get_documents(term)) + 0.5) / (len(self.get_documents(term)) + 0.5) + 1)

    def bm25(self, doc_id: int, term:str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)

        return tf * idf

    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
        query_tokens = tokenize_text(query)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results
        
        
    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter(tokens)
        for token in set(tokens):
            self.index[token].add(doc_id)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def load(self):
        try:
            with open(self.index_path, 'rb') as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, 'rb') as f:
                self.docmap = pickle.load(f)
            with open(self.term_freq_path, 'rb') as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, 'rb') as f:
                self.doc_lengths = pickle.load(f)
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

def tf_command(doc_id: int, term:str) -> int:
    idx = InvertedIndex()
    token_term = tokenize_word(term)
    
    try:
        idx.load()
    except Exception as e:
        print(f"Error loading  index and docmap: {e}")
        sys.exit(1)

    return idx.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f'Error loading index and docmap: {e}')

    token_term = tokenize_word(term)

    return  idx.get_idf(token_term)

def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f"Error loading index and docmap: {e}")
    tf = idx.get_tf(doc_id, term)
    idf = idx.get_idf(term)

    return tf * idf

def bm25_idf_command(term:str) -> float:
    idx = InvertedIndex()
    term = tokenize_word(term)
    try:
        idx.load()
    except Exception as e:
        print(f"Error loading index and docmap: {e}")

    return idx.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print(f"Error loading index: {e}")

    token = tokenize_word(term)
    bm25_tf = idx.get_bm25_tf(doc_id, token, k1, b)
    return bm25_tf

def bm25_search_command(query: str, limit: int = 5) -> dict[int: int]:
    idx = InvertedIndex()
    try:
        idx.load()
    except Exception as e:
        print("Error loading index: {e}")
    
    results = idx.bm25_search(query, limit)
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

def tokenize_word(text: str) -> str:
    text = tokenize_text(text)
    if len(text) != 1:
        raise Exception("Tokenizer didn't return just one word")
    return text[0]

