from ast import List
from typing import Dict, List, Optional, Union
from typing_extensions import Literal, TypedDict, NotRequired
from pydantic import BaseModel, Field


class CompletionRequestMessage(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = Field(
        default="user", description="The role of the message."
    )
    content: Optional[str] = Field(
        default="", description="The content of the message."
    )


class ChatCompletionRequestResponseFormat(TypedDict):
    type: Literal["text", "json_object"]


class RoleMapping(TypedDict):
    system_prompt: str
    system: str
    user: str
    assistant: str
    stop: str


class ChatCompletionRequest(BaseModel):
    messages: List[CompletionRequestMessage] = Field(
        default=[], description="A list of messages to generate completions for."
    )

    max_tokens: Optional[int] = Field(
        default=100,
        description="The maximum number of tokens to generate. Defaults to inf",
    )
    temperature: float = 0.6
    stop: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="A list of tokens at which to stop generation. If None, no stop tokens are used.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the results as they are generated. Useful for chatbots.",
    )

    seed: Optional[int] = Field(None)

    response_format: Optional[ChatCompletionRequestResponseFormat] = Field(
        default=None,
    )

    # custom fields
    role_mapping: Optional[RoleMapping] = Field(
        default=None,
        description="A dictionary mapping roles to their respective prefixes in the chat.",
    )


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseMessage(TypedDict):
    content: Optional[str]
    role: Literal["assistant"]


class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: "ChatCompletionResponseMessage"
    finish_reason: Optional[str]


class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List["ChatCompletionResponseChoice"]
    usage: CompletionUsage


class ChatCompletionStreamResponseDeltaEmpty(TypedDict):
    pass


class ChatCompletionStreamResponseDelta(TypedDict):
    content: NotRequired[str]
    role: NotRequired[Literal["system", "user", "assistant", "tool"]]


class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: Union[
        ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty
    ]
    finish_reason: Optional[Literal["stop", "length"]]


class CreateChatCompletionStreamResponse(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionStreamResponseChoice]
