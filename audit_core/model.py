"""Replaceable model transport; importing audit_core does not load an SDK."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelRequest:
    model: str
    messages: list[dict[str, str]]
    response_format: dict
    temperature: float = 0
    max_tokens: int = 512
    timeout: float = 30


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = "stop"
    refusal: str | None = None


class ModelBackend(Protocol):
    """One request, no hidden retries. Return observed usage, never invented zeros."""

    def complete(self, request: ModelRequest) -> ModelResponse: ...


class LiteLLMBackend:
    """Optional transport using the project's existing LiteLLM dependency.

    Requires a provider supporting strict JSON-schema responses. No fallback
    request or automatic retry is made if the provider rejects the request.
    """

    def complete(self, request: ModelRequest) -> ModelResponse:
        from litellm import completion

        response = completion(
            model=request.model, messages=request.messages,
            response_format=request.response_format, temperature=request.temperature,
            max_tokens=request.max_tokens, timeout=request.timeout, num_retries=0,
        )
        choice = response.choices[0]
        usage = response.usage
        return ModelResponse(
            content=choice.message.content,
            input_tokens=usage.prompt_tokens if usage is not None else None,
            output_tokens=usage.completion_tokens if usage is not None else None,
            finish_reason=choice.finish_reason,
            refusal=getattr(choice.message, "refusal", None),
        )
