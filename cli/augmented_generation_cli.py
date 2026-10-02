import argparse

from lib.augmented_generation import rag_command, summarize_command, citation_command, question_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Use LLM to sumarize search results"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG Summary")

    citation_parser = subparsers.add_parser("citations", help="Use RRF and cite sources (list what documents the results came from)")
    citation_parser.add_argument("query", type=str, help="User query to search")
    citation_parser.add_argument("limit", nargs="?", default=5, help="maximum number of documents to return in search")

    question_parser = subparsers.add_parser("question", help="Answer a user question using RAG")
    question_parser.add_argument("query", type=str, help="User query to search")
    question_parser.add_argument("limit", nargs="?", default=5, help="Limit the number of documents to return")

    args = parser.parse_args()

    match args.command:
        case "rag":
            result = rag_command(args.query)
            print("Search Results:")
            for document in result["search_results"]:
                print(f"  - {document['doc']['title']}")
            print()
            print("RAG Response:")
            print(result["answer"])
        case "summarize":
            result = summarize_command(args.query)
            print("Search Results:")
            for document in result['search_results']:
                print(f"  - {document['doc']['title']}")
            print()
            print("LLM Summary")
            print(result["answer"])
        case "citations":
            result = citation_command(args.query, args.limit)
            print("Search Results:")
            for document in result['search_results']:
                print(f" - {document['doc']['title']}")
            print("LLM Answer")
            print(result['answer'])
        case "question":
            result = question_command(args.query, args.limit)
            print("Search Results:")
            for document in result['search_results']:
                print(f" - {document['doc']['title']}")
            print("Answer:")
            print(result['answer'])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
