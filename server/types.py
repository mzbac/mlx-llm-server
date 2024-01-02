from ast import List
from typing import List, Optional, Union
from typing_extensions import TypedDict, Literal
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
