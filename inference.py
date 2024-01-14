import argparse
from mlx_lm import load, generate

DEFAULT_PROMPT = "Hello, world!"
DEFAULT_MAX_TOKENS = 100
DEFAULT_TEMP = 0.7

def main():
    parser = argparse.ArgumentParser(description="Generate text using a language model")

    parser.add_argument(
        "--model",
        type=str,
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_TEMP,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    model, tokenizer = load(args.model)
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        temp=args.temp,
        max_tokens=args.max_tokens,
        verbose=True
    )

    print(response)

if __name__ == "__main__":
    main()
