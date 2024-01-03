from threading import Lock
import time
from typing import List, Optional, Tuple
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from .types import (
    ChatCompletionRequest,
    CompletionRequestMessage,
    CompletionUsage,
    CreateChatCompletionResponse,
    CreateChatCompletionStreamResponse,
)
from sse_starlette.sse import EventSourceResponse
from pathlib import Path
import json
import uuid
from transformers import AutoTokenizer
from mlx.utils import tree_unflatten
import mlx.core as mx
import mlx.nn as nn
from llm.llama.llama import Llama, ModelArgs
import numpy as np

_llama_model: Optional[Llama] = None

llama_outer_lock = Lock()
llama_inner_lock = Lock()


def set_llama_model(model: Llama):
    global _llama_model
    _llama_model = model


def get_llama_model():
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield _llama_model
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()


def load_model(model_path: str, disable_fast_tokenizer: bool):
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
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def is_stop_condition_met(
    tokens: List[int],
    stop_id_sequences: List[np.ndarray],
    eos_token_id: int,
    max_tokens: int,
) -> Tuple[bool, bool]:
    """
    Determines if the token generation should stop based on various conditions.

    Args:
    - tokens: List of generated token IDs.
    - stop_id_sequences: List of token ID sequences that trigger a stop.
    - eos_token_id: Token ID indicating the end of a sequence.
    - max_tokens: Maximum number of tokens to generate.

    Returns:
    - A tuple (stop_met, trim_needed) where:
      - stop_met (bool): True if the stop condition is met, False otherwise.
      - trim_needed (bool): True if the generated tokens need to be trimmed, False otherwise.
    """

    if len(tokens) >= max_tokens:
        return True, False

    if tokens and tokens[-1] == eos_token_id:
        return True, True

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if np.all(np.equal(np.array(tokens[-len(stop_ids) :]), stop_ids)):
                return True, True

    return False, False


def generate(
    prompt: mx.array,
    model: Llama,
    temp: float = 0.0,
    stop_id_sequences: List[np.ndarray] = None,
    eos_token_id: int = None,
    max_tokens: int = 100,
):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            return mx.random.categorical(logits * (1 / temp))

    y = prompt
    cache = None
    tokens = []

    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        token = y.item()
        tokens.append(token)

        stop_met, trim_needed = is_stop_condition_met(
            tokens, stop_id_sequences, eos_token_id, max_tokens
        )
        if stop_met:
            if trim_needed:
                if stop_id_sequences:
                    tokens = tokens[
                        : -len(stop_id_sequences[0])
                    ]  # Trim the stop sequence
                else:
                    tokens.pop()  # Remove the last token (EOS)
            break

        yield token


def convert_chat(
    messages: CompletionRequestMessage, role_mapping: Optional[dict] = None
):
    default_role_mapping = {
        "system_prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant follows the given rules no matter what.",
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line.role, "")
        stop = role_mapping.get("stop", "")
        prompt += f"{role_prefix}{line.content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


def create_app(model_path: str, disable_fast_tokenizer: bool):
    model, tokenizer = load_model(model_path, disable_fast_tokenizer)
    set_llama_model(model)
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        _: Request,
        body: ChatCompletionRequest,
        model: Llama = Depends(get_llama_model),
    ):
        chat_id = f"chatcmpl-{uuid.uuid4()}"
        prompt = convert_chat(body.messages, body.role_mapping)
        prompt = tokenizer(
            prompt,
            return_tensors="np",
            return_attention_mask=False,
        )[
            "input_ids"
        ][0]
        prompt = mx.array(prompt)
        stop_words = body.stop if body.stop else []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            tokenizer(stop_word, return_tensors="np", add_special_tokens=False)[
                "input_ids"
            ][0]
            for stop_word in stop_words
        ]
        eos_token_id = tokenizer.eos_token_id
        max_tokens = body.max_tokens

        if body.stream:

            async def event_generator():
                for token in generate(
                    prompt,
                    model,
                    body.temperature,
                    stop_id_sequences,
                    eos_token_id,
                    max_tokens,
                ):
                    s = tokenizer.decode(token)
                    response = CreateChatCompletionStreamResponse(
                        id=chat_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model="gpt-3.5-turbo",
                        system_fingerprint=f"fp_{uuid.uuid4()}",
                        choices=[
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": s},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    )

                    yield f"{json.dumps(response)}"

            return EventSourceResponse(event_generator())
        else:
            tokens = list(
                generate(
                    prompt,
                    model,
                    body.temperature,
                    stop_id_sequences,
                    eos_token_id,
                    max_tokens,
                )
            )
            s = tokenizer.decode(tokens)
            response = CreateChatCompletionResponse(
                id=chat_id,
                object="chat.completion",
                created=int(time.time()),
                model="gpt-3.5-turbo",
                system_fingerprint=f"fp_{uuid.uuid4()}",
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": s},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
                usage=CompletionUsage(
                    prompt_tokens=len(prompt),
                    completion_tokens=len(tokens),
                    total_tokens=len(prompt) + len(tokens),
                ),
            )
            return f"{json.dumps(response)}"

    return app
