import argparse
from pathlib import Path
import json
from transformers import AutoTokenizer
from mlx.utils import tree_unflatten
import mlx.core as mx
import mlx.nn as nn
from llm.llama.llama import Llama, ModelArgs


def generate(
    prompt: mx.array,
    model: Llama,
    temp: float = 0.0,
):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        yield y


def load_model(model_path: str, disable_fast_tokenizer:bool):
    model_path = Path(model_path)
    with open(model_path / "config.json", "r") as f:
        config = json.load(f)
        config.pop("model_type")
        quantization = config.pop("quantization", None)
        model_args = ModelArgs(**config)

    model = Llama(model_args)
    weights = mx.load(str(model_path / "weights.npz"))
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    model.update(tree_unflatten(list(weights.items())))

    if disable_fast_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False
    )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="llm inference script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the mlx model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="hello",
    )
    parser.add_argument(
        "--disable-fast-tokenizer",
        "-dft",
        action="store_true",
        help="Disable the fast tokenizer",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.6,
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(args.model_path, args.disable_fast_tokenizer)

    prompt = tokenizer(
        args.prompt,
        return_tensors="np",
        return_attention_mask=False,
    )[
        "input_ids"
    ][0]

    prompt = mx.array(prompt)

    print(args.prompt, end="", flush=True)

    tokens = []
    skip = 0
    for token, _ in zip(
        generate(prompt, model, args.temp),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        print(s[skip:], end="", flush=True)
        skip = len(s)

    print(tokenizer.decode(tokens)[skip:], flush=True)
