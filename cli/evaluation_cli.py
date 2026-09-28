import argparse
from lib.search_utils import GOLDEN_DATASET_PATH
import json
import os
from lib.hybrid_search import rrf_search

def main() -> None:
     parser = argparse.ArgumentParser(description="Search Evaluation CLI")
     parser.add_argument(
         "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

     args = parser.parse_args()
     limit = args.limit

     # run evaluation logic here
     with open(GOLDEN_DATASET_PATH, 'r') as f:
        tests = json.load(f)['test_cases']

     print(f"k={limit}")
     for test in tests:
        query = test['query']
        relevant_docs = set(test['relevant_docs'])
        results = rrf_search(query, 60, limit)
        titles = set(doc.get('doc').get('title') for doc in results)
        docs_found = len(relevant_docs.intersection(titles))
        precision = round(docs_found / len(results), 4)

        print(f"""- Query: {query}\n  - Precision@{limit}: {precision}\n  - Retrieved: {' '.join(titles)}\n  - Relevant: {' '.join(relevant_docs)}""")


if __name__ == "__main__":
     main()
