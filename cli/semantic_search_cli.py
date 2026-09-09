import argparse
from lib.semantic_search import(
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query,
    search_command,
    chunk_command,
    semantic_chunk
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser(
        "verify", help="Verify semantic model has loaded and see max sequence length"
    )
    embed_parser = subparsers.add_parser(
        "embed_text", help="Create word embeddings for semantic search"
    )
    embed_parser.add_argument("text", type=str, help="Text to embed")
    
    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify document embeddings using semantic model"
    )
    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Embed a query so we can compare it to our document embeddings"
    )
    embed_query_parser.add_argument("query", type=str, help="User query to embed")

    search_parser = subparsers.add_parser("search", help="Use semantic search to find similar movies")
    search_parser.add_argument("query", type=str, help="User query to search")
    search_parser.add_argument("--limit", nargs="?", default=5)

    chunk_parser = subparsers.add_parser(
            "chunk", help="Chunk text to improve ability to find good semantic matches."
            )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", nargs="?", default=200, type=int)
    chunk_parser.add_argument("--overlap", nargs="?", default=0, type=int)

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Semantically chunk text given an overalp and chunk size"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", nargs="?", default=4, type=int)
    semantic_chunk_parser.add_argument("--overlap", nargs="?", default=0, type=int)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "verify_embeddings":
            verify_embeddings()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query(args.query)
        case "search":
            search_command(args.query, int(args.limit))
        case "chunk":
            results = chunk_command(args.text, int(args.chunk_size), int(args.overlap))
            print(f"Chunking {len(args.text)} characters")
            for i, result in enumerate(results):
                print(f"{i+1}. {result}")
        case "semantic_chunk":
            results = semantic_chunk(args.text, int(args.max_chunk_size), int(args.overlap))
            print(f"Semantically chunking {len(args.text)} characters")
            for i, result in enumerate(results):
                print(f"{i+1}. {result}")
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
