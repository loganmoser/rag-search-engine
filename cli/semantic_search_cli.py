import argparse
from lib.semantic_search import(
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query,
    search_command
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
        case _:
            parser.print_help()




if __name__ == "__main__":
    main()
