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
import json
import uuid
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

_llama_model: Optional[nn.Module] = None

llama_outer_lock = Lock()
llama_inner_lock = Lock()


def set_llama_model(model: nn.Module):
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


def is_stop_condition_met(
    tokens: List[int],
    stop_id_sequences: List[np.ndarray],
    eos_token_id: int,
    max_tokens: int,
) -> Tuple[bool, bool, int]:
    if len(tokens) >= max_tokens:
        return True, False, 0

    if tokens and tokens[-1] == eos_token_id:
        return True, True, 1

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if np.all(np.equal(np.array(tokens[-len(stop_ids) :]), stop_ids)):
                return True, True, len(stop_ids)

    return False, False, 0


def generate(
    prompt: mx.array,
    model: nn.Module,
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

        stop_met, trim_needed, trim_length = is_stop_condition_met(
            tokens, stop_id_sequences, eos_token_id, max_tokens
        )
        if stop_met:
            if trim_needed and trim_length > 0:
                tokens = tokens[:-trim_length]
            tokens = None
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


def create_app(model_path: str):
    model, tokenizer = load(model_path)
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
        model=Depends(get_llama_model),
    ):
        chat_id = f"chatcmpl-{uuid.uuid4()}"
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                body.messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="np",
            )
        else:
            prompt = convert_chat(body.messages, body.role_mapping)
            prompt = tokenizer.encode(prompt, return_tensors="np")
        prompt = mx.array(prompt[0])
        stop_words = body.stop if body.stop else []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            tokenizer.encode(stop_word, return_tensors="np", add_special_tokens=False)[
                0
            ]
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
