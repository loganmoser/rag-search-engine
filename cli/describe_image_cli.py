import argparse
import os
from dotenv import load_dotenv
from openai import OpenAI
import mimetypes
import pathlib
import base64


def main():
    # Parse user arguments
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--image", type=pathlib.Path, help="Path to image file")
    parser.add_argument("--query", type=str, help="User text query to rewrite based on the image")

    args = parser.parse_args()

    # Get MIME type of file to feed into AI model - default to image/jpeg if None
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"


    # Convert image to readable byte format for model
    with open(args.image, 'rb') as f:
        img = f.read()

    # Creat OpenAI client using openrouter
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    model = "openrouter/free"

    # Give system prompt for analyzing image and rewriting user query
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    # Base64 encod the image so it is readable by openrouter
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": args.query.strip()},
            ],
        }
    ]

    # Print response from openrotuer for user to see
    response = client.chat.completions.create(
        model=model, messages=messages
    )
    content = response.choices[0].message.content
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens: {response.usage.total_tokens}")



if __name__ == "__main__":
    main()
