# RAG Search Engine

A command-line search engine built in Python that combines classic keyword search with modern semantic (embedding-based) search, then layers on hybrid ranking, multimodal (image + text) search, and Retrieval-Augmented Generation (RAG) using an LLM.

Built as a certification project for the [Boot.dev](https://www.boot.dev) RAG module.

## Features

- **Keyword search**: inverted index with stemming and stop-word removal, ranked with TF-IDF / BM25
- **Semantic search**: finds results by *meaning* using sentence embeddings, not just matching words
- **Hybrid search**: merges keyword and semantic results (weighted scoring or Reciprocal Rank Fusion)
- **Multimodal search**: search with images and text using a CLIP-style model
- **RAG**: feeds the top search results to an LLM to generate grounded answers, summaries, and citations
- **Caching**: indexes and embeddings are built once and stored in `cache/` so repeat searches are fast

## Project Structure

```
rag-search-engine/
├── cli/              # Command-line entry points and search logic
├── cache/            # Generated indexes and embeddings (git-ignored)
├── test_llm.py       # Quick check that your LLM API connection works
├── pyproject.toml    # Project metadata and dependencies
└── uv.lock           # Locked dependency versions
```

## Tech Stack

| Tool | Used for |
|------|----------|
| Python 3.13+ | Language |
| [NLTK](https://www.nltk.org/) | Tokenizing and stemming for keyword search |
| [NumPy](https://numpy.org/) | Vector math (cosine similarity, score normalization) |
| [sentence-transformers](https://sbert.net/) | Text and image embeddings |
| [Pillow](https://python-pillow.org/) | Loading images for multimodal search |
| [OpenAI SDK](https://github.com/openai/openai-python) | LLM calls for RAG features |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Loading API keys from a `.env` file |
| [uv](https://docs.astral.sh/uv/) | Dependency and environment management |

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/loganmoser/rag-search-engine.git
cd rag-search-engine
uv sync
```

### 2. Add your API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY="your_key_here"
```

> Note: if you're using a different provider or key name, update this to match your code.

### 3. Verify the LLM connection (optional)

```bash
uv run python test_llm.py
```

## Usage

> **TODO:** The commands and flags below are examples of the pattern. Replace the script names, subcommands, and arguments with the exact ones in your `cli/` folder.

### Keyword search

Finds documents that contain your query terms, ranked by BM25.

```bash
# Build the inverted index first (only needed once)
uv run cli/keyword_search_cli.py build

# Search
uv run cli/keyword_search_cli.py search "space adventure"
```

### Semantic search

Finds documents with similar *meaning*, even when the words differ.

```bash
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py search "a story about robots learning to feel" --limit 5
```

### Hybrid search

Combines both approaches. Weighted mode blends normalized scores; RRF mode blends rankings.

```bash
uv run cli/hybrid_search_cli.py weighted-search "heist movie with a twist" --alpha 0.5
uv run cli/hybrid_search_cli.py rrf-search "heist movie with a twist" --k 60
```

### Multimodal search

Search using an image (or an image plus text).

```bash
uv run cli/multimodal_search_cli.py image_search path/to/image.jpg
```

### RAG (answers from your data)

Retrieves the top results, then asks an LLM to answer using only those results.

```bash
uv run cli/augmented_generation_cli.py rag "What are some good family-friendly adventure movies?"
uv run cli/augmented_generation_cli.py summarize "space exploration" --limit 5
```

Run any script with `--help` to see every available command and argument:

```bash
uv run cli/hybrid_search_cli.py --help
```

## How It Works

1. **Index.** Documents are tokenized, stemmed, and stored in an inverted index (word → documents that contain it). Separately, each document is converted into an embedding vector that captures its meaning.
2. **Retrieve.** A query is run through keyword scoring (BM25), semantic scoring (cosine similarity between embeddings), or both.
3. **Combine.** Hybrid search normalizes and merges the two result lists so you get exact-match precision *and* meaning-based recall.
4. **Generate.** In RAG mode, the top results are inserted into a prompt so the LLM answers from real retrieved context instead of guessing.

## Learn the Concepts

Resources for understanding the ideas behind each piece:

- **TF-IDF:** [Wikipedia: tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), a good starting point for how word importance is scored
- **BM25:** [Wikipedia: Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25), the ranking function behind most keyword search engines
- **Stemming and tokenization:** [NLTK Book, Chapter 3](https://www.nltk.org/book/ch03.html)
- **Embeddings and semantic search:** [Sentence Transformers docs](https://sbert.net/) (see the semantic search section)
- **Cosine similarity:** [Wikipedia: Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- **RAG overview:** [Boot.dev](https://www.boot.dev) RAG module (the course this project came from)

## What I Learned

- Why keyword search misses synonyms and how embeddings solve that
- Why embeddings alone can miss exact terms, which is the case for hybrid search
- How to normalize scores from different systems so they can be combined fairly
- Caching expensive computation (embeddings) to disk
- Building a multi-command CLI with `argparse` subparsers

## License

Add a license of your choice (e.g., MIT) or remove this section.
