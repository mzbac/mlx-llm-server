import argparse
import uvicorn
import os

from mlx_llm_server.app import create_app

def main():
    parser = argparse.ArgumentParser(description="mlx llama python server.")
    parser.add_argument(
        "--model-path",
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
    app = create_app(args.model_path, args.adapter_file)
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", 8080)),
    )
    
if __name__ == "__main__":
    main()
