import argparse
from lib.keyword_search import search_command, InvertedIndex, build_command, tf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build InvertedIndex for TF-IDF")

    term_parser = subparsers.add_parser("tf", help="Get term frequencies from a document for a term")
    term_parser.add_argument("doc_id", type = int, help="document_id for tf")
    term_parser.add_argument("term", type=str, help="term to search within a document")

    args = parser.parse_args()

    print(args)

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ID:{res['id']} {res['title']}")
        case "tf":
            tf_command(args.doc_id, args.term)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
