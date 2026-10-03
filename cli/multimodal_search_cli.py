import argparse
import pathlib
from lib.multimodal_search import verify_image_embedding, image_serach_command

def main() -> None:
    # Parse user arguments
    parser = argparse.ArgumentParser(description="Image embedding CLI tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify Image Embeddings"
    )
    image_parser.add_argument("image", type=pathlib.Path, help="Path to image file")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search movie database with provided image file path"
    )
    image_search_parser.add_argument("image_path", type=pathlib.Path, help="Image filepath for query")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        case "image_search":
            results = image_serach_command(args.image_path)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (similarity: {result['similarity_score']:.3f})")
                print(f"  {result['description'][:100]}...")

if __name__ == "__main__":
    main()
