import argparse
from lib.keyword_search import (
        search_command, 
        build_command, 
        tf_command, 
        idf_command, 
        tfidf_command, 
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build InvertedIndex for TF-IDF")

    term_parser = subparsers.add_parser("tf", help="Get term frequencies from a document for a term")
    term_parser.add_argument("doc_id", type = int, help="document_id for tf")
    term_parser.add_argument("term", type=str, help="term to search within a document")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get Term Frequency * inverse document frequency for term")
    tfidf_parser.add_argument("doc_id", type=int, help="document id to search")
    tfidf_parser.add_argument("term", type=str, help="Term to get tf-idf for")

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
            tf = tf_command(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
