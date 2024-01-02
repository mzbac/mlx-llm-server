import argparse
import copy
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.llama.config import ModelArgs
from llm.llama.llama import Llama


def quantize(weights, config, args):
    quantized_config = copy.deepcopy(config)

    # Load the model:
    model_args = ModelArgs(**config)
    model = Llama(model_args)

    weights = tree_map(mx.array, weights)
    model.update(tree_unflatten(list(weights.items())))

    # Quantize the model:
    nn.QuantizedLinear.quantize_module(model, args.q_group_size, args.q_bits)

    # Update the config:
    quantized_config["quantization"] = {
        "group_size": args.q_group_size,
        "bits": args.q_bits,
    }
    quantized_weights = dict(tree_flatten(model.parameters()))

    return quantized_weights, quantized_config


def convert(args):
    hf_path = Path(args.hf_path)

    model = AutoModelForCausalLM.from_pretrained(
        str(hf_path), torch_dtype=torch.float16
    )
    config = model.config.to_dict()

    state_dict = model.state_dict()
    if args.disable_fast_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(str(hf_path), use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(hf_path))

    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    weights = {k: mx.array(v.numpy(), mx.bfloat16) for k, v in state_dict.items()}

    return weights, config, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert llama model to npz")
    parser.add_argument(
        "--hf-path",
        help="The huggingface model to be converted",
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        default="mlx_model",
        help="The path to save the MLX model.",
    )
    parser.add_argument(
        "--disable-fast-tokenizer",
        "-dft",
        action="store_true",
        help="Disable the fast tokenizer",
    )
    parser.add_argument(
        "-q",
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--q-group-size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--q-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )
    args = parser.parse_args()

    mlx_path = Path(args.mlx_path)
    mlx_path.mkdir(parents=True, exist_ok=True)

    weights, config, tokenizer = convert(args)

    if args.quantize:
        print("[INFO] Quantizing")
        weights, config = quantize(weights, config, args)

    np.savez(str(mlx_path / "weights.npz"), **weights)
    tokenizer.save_pretrained(mlx_path)
    with open(mlx_path / "config.json", "w") as f:
        config["model_type"] = "llama"
        json.dump(config, f, indent=4)
