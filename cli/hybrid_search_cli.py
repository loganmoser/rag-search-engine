import argparse

from lib.hybrid_search import (
    normalize_scores,
    weighted_search,
    rrf_search,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Combine BM25 and Semanntic search by min/max normlaizing results")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Combine BM25 and Semantic search results")
    weighted_search_parser.add_argument("query", type=str, help="Query to score")
    weighted_search_parser.add_argument("--alpha", nargs="?", type=float, default=0.5, help="dynamically controls the weighting of the two scores")
    weighted_search_parser.add_argument("--limit", nargs="?", type=int, default=5)

    rff_search_parser = subparsers.add_parser("rrf-search", help="Use Reverse Rank Fusion Search")
    rff_search_parser.add_argument("query", type=str, help="Query to score")
    rff_search_parser.add_argument("-k", nargs="?", type=int, default=60, help="Constant used to scale rankings. 60 by default")
    rff_search_parser.add_argument("--limit", nargs="?", type=int, default=5, help="Number of results to return. Defaults to 5")
    rff_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite"], help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
