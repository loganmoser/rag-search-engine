import os

from dotenv import load_dotenv
from openai import OpenAI

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    RRF_K,
    SEARCH_MULTIPLIER,
    load_movies,
)

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"


def generate_answer(search_results, query, llm_task: str | None = None, limit: int=5):
    context = ""

    for result in search_results[:limit]:
        context += f"{result['doc']['title']}: {result['doc']['document']}\n\n"

    match llm_task:
        case "summarize":
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

            This should be tailored to Webflyx users. Webflyx is a movie streaming service.

            Query: {query}

            Search results:
            {search_results}

            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
        case _:
            prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {context}

            Answer:"""

    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return (response.choices[0].message.content or "").strip()


def rag(query, llm_task: str | None = None, limit=DEFAULT_SEARCH_LIMIT):
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    search_results = hybrid_search.rrf_search(
        query, k=RRF_K, limit=limit * SEARCH_MULTIPLIER
    )

    if not search_results:
        return {
            "query": query,
            "search_results": [],
            "error": "No results found",
        }

    answer = generate_answer(search_results, query, llm_task, limit)

    return {
        "query": query,
        "search_results": search_results[:limit],
        "answer": answer,
    }

def summarize_command(query, llm_task: str = "summarize"):
    return rag(query, llm_task)

def rag_command(query):
    return rag(query)
