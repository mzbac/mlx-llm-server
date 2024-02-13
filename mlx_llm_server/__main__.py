import argparse
import os

from mlx_llm_server.app import run


def main():
    parser = argparse.ArgumentParser(description="mlx llama python server.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the mlx model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        help="The path for the trained adapter weights.",
    )
    args = parser.parse_args()
    run(
        os.getenv("HOST", "127.0.0.1"),
        int(os.getenv("PORT", 8080)),
        args.model,
        args.adapter_file,
    )


if __name__ == "__main__":
    main()
