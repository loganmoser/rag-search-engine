import os
from dotenv import load_dotenv
from openai import OpenAI

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

from .search_utils import(
    Movie,
    load_movies,
)


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        #Takes BM25 scores and semantic search scores, normalizes them based on the alpha provided, then returns the top *limit* scores
        big_limit = limit * 500
        keyword_results = self._bm25_search(query, big_limit)
        semantic_results = self.semantic_search.search_chunks(query, big_limit)
        #Build total scores for individual movies
        best_semantic_scores = {}
        for doc in semantic_results:
            id = doc['id']
            if id not in best_semantic_scores:
                best_semantic_scores[id] = doc
            else:
                if doc['score'] > best_semantic_scores[id]['score']:
                    best_semantic_scores[id] = doc
        #Normalize both scores using min/max function
        bm_normalize = normalize_scores([kw['score'] for kw in keyword_results])
        semantic_normalized = normalize_scores([sm['score'] for sm in best_semantic_scores.values()])
        scores_map = {}
        # Iterate through keyword searched building a dicionary entry with a 0 for the semantic score
        # If the semantic score is found on the second loop, replace it. Otherwise, build a new entry
        # with 0 for keyword score.
        for doc, score in zip(keyword_results, bm_normalize):
            result = {
                'title': doc['title'],
                'desc': doc['document'],
                'keyword_score': score,
                'semantic_score': 0.0
            }
            scores_map[doc['id']] = result
        for doc, score in zip(best_semantic_scores.values(), semantic_normalized):
            if doc['id'] not in scores_map:
                result = {
                    'title': doc['title'],
                    'desc': doc['document'],
                    'keyword_score': 0.0,
                    'semantic_score': score
                }
                scores_map[doc['id']] = result
            else:
                scores_map[doc['id']]['semantic_score'] = score


        for doc in scores_map.values():
            doc['hybrid_score'] = hybrid_score(doc['keyword_score'], doc['semantic_score'], alpha)

        sorted_scores = sorted(scores_map.values(), key= lambda x: x['hybrid_score'], reverse=True)
        return sorted_scores[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        big_limit = limit * 500
        bm25_results = self._bm25_search(query, big_limit)
        semantic_results = self.semantic_search.search_chunks(query, big_limit)
        results_map = {}
        for i,doc in enumerate(bm25_results,1):
            id = doc['id']
            result = {
                "doc": doc,
                "bm25_rank": i,
                "semantic_rank": None
            }
            results_map[id] = result
        for i, doc in enumerate(semantic_results, 1):
            id = doc['id']
            if id in results_map:
                results_map[id]['semantic_rank'] = i
            else:
                result = {
                    "doc": doc,
                    "bm25_rank": None,
                    "semantic_rank": i
                }
        
        for movie in results_map.values():
            if movie['bm25_rank'] and movie['semantic_rank']:
                movie['rrf_score'] = rff_score(movie['bm25_rank'], k) + rff_score(movie['semantic_rank'], k)
            elif movie['bm25_rank'] and not movie['semantic_rank']:
                movie['rrf_score'] = rff_score(movie['bm25_rank'], k)
            else:
                movie['rrf_score'] = rff_score(movie['semantic_rank'], k)

        sorted_results = sorted(results_map.values(), key = lambda x: x['rrf_score'], reverse=True)[:limit]

        return sorted_results
        

def rrf_search(query: str, k: int = 60, limit: int = 5, enhance: str = None):
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url = "https://openrouter.ai/api/v1",
        api_key=api_key
    )

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    match enhance:
        case "spell":
            messages = [
                {
                    "role": "user",
                    "content":f"""Fix any spelling errors in the user-provided movie search query below.
                                Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                                Preserve punctuation and capitalization unless a change is required for a typo fix.
                                If there are no spelling errors, or if you're unsure, output the original query unchanged.
                                Output only the final query text, nothing else.
                                User query: "{query}"
                                """
                }
            ]

            response = client.chat.completions.create(
                model='openrouter/free',
                messages=messages
            )
            enhanced_query = response.choices[0].message.content
            
            print(f"Enhanced query ({enhance}: '{query}' -> '{enhanced_query}')\n")

            results = hybrid_search.rrf_search(enhanced_query, k, limit)
            for i, result in enumerate(results, 1):
                print(f"{i}.  {result['doc']['title']}\n  RRF Score: {result['rrf_score']}\n  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n  {result['doc']['document'][:50]}")
        case "rewrite":
            messages = [
                {
                    "role": "user",
                    "content": f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep the rewritten query concise (under 10 words)
                        - It should be a Google-style search query, specific enough to yield relevant results
                        - Don't use boolean logic

                        Examples:
                        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                        If you cannot improve the query, output the original unchanged.
                        Output only the rewritten query text, nothing else.

                        User query: "{query}"
"""               }
            ]

            response = client.chat.completions.create(
                model='openrouter/free',
                messages=messages
            )
            enhanced_query = response.choices[0].message.content
            
            print(f"Enhanced query ({enhance}: '{query}' -> '{enhanced_query}')\n")

            results = hybrid_search.rrf_search(enhanced_query, k, limit)
            for i, result in enumerate(results, 1):
                print(f"{i}.  {result['doc']['title']}\n  RRF Score: {result['rrf_score']}\n  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n  {result['doc']['document'][:50]}")

        case _:
            results = hybrid_search.rrf_search(query, k, limit)
            for i, result in enumerate(results, 1):
                print(f"{i}.  {result['doc']['title']}\n  RRF Score: {result['rrf_score']}\n  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n  {result['doc']['document'][:50]}")
   
def weighted_search(query: str, alpha: float, limit: int = 5) -> None:
    documents = load_movies()
    hybrid_search = HybridSearch(documents)
    results = hybrid_search.weighted_search(query, alpha, limit)
    for i, doc in enumerate(results, 1):
        print(f"""{i}. {doc['title']}\n  Hybrid Score: {doc['hybrid_score']:.3f}\n  BM25: {doc['keyword_score']:.3f}, Semantic: {doc['semantic_score']:.3f}\n  {doc['desc'][:50]}\n""")


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    normalized_scores = []
    for s in scores:
        normalized_scores.append((s - min_score) / (max_score - min_score))
    return normalized_scores

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1-alpha) * semantic_score

def rff_score(rank:int, k:int = 60) -> float:
    return 1 / (k + rank)
