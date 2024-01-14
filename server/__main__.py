import argparse
import uvicorn
import os

from server.app import create_app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mlx llama python server.")
    parser.add_argument(
        "--model-path",
        type=str,
        help="The path to the mlx model weights, tokenizer, and config",
    )
    args = parser.parse_args()
    app = create_app(args.model_path)
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8080)),
    )
