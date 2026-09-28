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
    rff_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rff_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"])

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            results = rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method)

            match args.rerank_method:
                case "individual":
                    print(f"Re-ranking the top {args.limit} results using {args.rerank_method} method...")
                    print(f"Reciprocal Rank Fusion Results for {args.query} (k={args.k})")
                    for i, doc in enumerate(results, 1):
                        doc_data = doc['doc'] # Adding the whole document to the list above means we need to get individual doc items here
                        print(f"""{i}. {doc_data['title']}
                                Re-rank score: {doc['rerank_score']}
                                RRF Score: {doc['rrf_score']}
                                BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}
                                {doc_data['document'][:50]}...""")
                case "batch":
                    for i, doc in enumerate(results, 1):
                        doc_data = doc['doc']
                        print(f"""{i}. {doc_data['title']}
                              Re-rank Rank: {i}
                              RRF Score: {doc['rrf_score']}
                              BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}
                              {doc_data['document'][:50]}...f""")
                case "cross_encoder":
                    print(f"Re-ranking top {limit} results using {rerank_method} method...")
                    print(f"Reciprocal Rank Fusion Results for {query} (k={k})")
                    for i, doc in enumerate(sorted_scores, 1):
                        print(f"""{i}. {doc.get('doc', '').get('title', '')}
                        Cross Encoder Score: {doc['cross_encoder_score']}
                        RRF Score: {doc['rrf_score']}
                        BM25 Rank: {doc['bm25_rank']}, Semantic Rank: {doc['semantic_rank']}
                        {doc['doc']['document'][:50]}...""")
                case _:
                    for i, result in enumerate(results, 1):
                        print(f"{i}.  {result['doc']['title']}\n  RRF Score: {result['rrf_score']}\n  BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}\n  {result['doc']['document'][:50]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
