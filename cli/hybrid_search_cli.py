import argparse

from lib.hybrid_search import (
    normalize_scores
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Combine BM25 and Semanntic search by min/max normlaizing results")
    normalize_parser.add_argument("scores", nargs="*", type=float, help="List of scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = normalize_scores(args.scores)
            for score in scores:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
