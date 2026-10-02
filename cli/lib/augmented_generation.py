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
        case "citation":
            prompt = f"""Answer the query below and give information based on the provided documents.

            The answer should be tailored to users of Webflyx, a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

            Query: {query}

            Documents:
            {search_results}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
        case "question":
            prompt = f"""Answer the user's question based on the provided movies that are available on Webflyx, a streaming service.

            Question: {query}

            Documents:
            {search_results}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
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

def question_command(query, limit, llm_task: str | None = "question"):
    return rag(query, "question", limit)

def citation_command(query, limit, llm_task: str | None = "citation"):
    return rag(query, "citation", limit)

def summarize_command(query, llm_task: str = "summarize"):
    return rag(query, llm_task)

def rag_command(query):
    return rag(query)
