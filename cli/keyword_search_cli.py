import argparse
from lib.keyword_search import search_command, InvertedIndex, build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build InvertedIndex for TF-IDF")

    args = parser.parse_args()
    
    inverted_index = InvertedIndex()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = search_command(args.query, inverted_index=inverted_index)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
