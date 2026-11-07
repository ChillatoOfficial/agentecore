import json
import logging
import os
from typing import Any

from agentecore.core.clients.client import ClientResponse
from agentecore.core.clients.http import HTTPClient
from agentecore.core.clients.stream import StreamAccumulator
from agentecore.type.type import (
    Block,
    FunctionCallBlock,
    FunctionCallResultBlock,
    Media,
    MediaBlock,
    Memory,
    Role,
    TextBlock,
)

logger = logging.getLogger(__name__)


def _loads(raw: str, name: str) -> dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        logger.warning("argomenti non validi per la tool call '%s': %r", name, raw)
        return {}


def _usage(data: dict[str, Any] | None) -> dict[str, int]:
    data = data or {}
    return {
        "prompt_tokens": data.get("prompt_tokens", 0),
        "completion_tokens": data.get("completion_tokens", 0),
    }


class OpenAI(HTTPClient):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        system_prompt: str = "",
        temperature: float | None = None,
        **http_options: Any,
    ):
        super().__init__(model_name, system_prompt, temperature, base_url=base_url, **http_options)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def _build_request(
        self,
        *,
        input: list[Block],
        tools: list[Any],
        memory: Memory,
        temperature: float | None,
        system_prompt: str,
        stream: bool = False,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        messages = self._to_messages(memory, input, system_prompt)
        payload: dict[str, Any] = {"model": self.model_name, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [self._to_openai_tool(t) for t in tools]
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}

        return (
            f"{self.base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}"},
            payload,
        )

    @staticmethod
    def _to_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
        # tool = {"name": ..., "description": ..., "parameters": <JSON Schema>}
        return {"type": "function", "function": tool}

    def _to_messages(
        self, memory: Memory, input: list[Block], system_prompt: str
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for turn in memory.turns:
            messages.extend(self._turn_to_messages(turn.role, turn.blocks))
        messages.extend(self._turn_to_messages(Role.USER, input))
        return messages

    @staticmethod
    def _media_part(media: Media) -> dict[str, Any]:
        if media.is_image:
            url = media.source if media.source_type == "url" else f"data:{media.mime_type};base64,{media.data_base64()}"
            return {"type": "image_url", "image_url": {"url": url}}
        if media.source_type == "url":
            raise ValueError("OpenAI accetta i PDF solo come dati: usa Media.from_path o Media.from_bytes")
        return {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": f"data:{media.mime_type};base64,{media.data_base64()}",
            },
        }

    @classmethod
    def _turn_to_messages(cls, role: Role, blocks: list[Block]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        calls = [b for b in blocks if isinstance(b, FunctionCallBlock)]
        has_media = any(isinstance(b, MediaBlock) for b in blocks)

        for b in blocks:
            if isinstance(b, FunctionCallResultBlock):
                messages.append({"role": "tool", "tool_call_id": b.id, "content": b.result})

        content: str | list[dict[str, Any]] | None
        if has_media:
            content = []
            for b in blocks:
                if isinstance(b, TextBlock):
                    content.append({"type": "text", "text": b.content})
                elif isinstance(b, MediaBlock) and b.media is not None:
                    content.append(cls._media_part(b.media))
        else:
            content = "".join(b.content for b in blocks if isinstance(b, TextBlock)) or None

        if content or calls:
            message: dict[str, Any] = {"role": role.value, "content": content}
            if calls:
                message["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in calls
                ]
            messages.append(message)
        return messages

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ClientResponse:
        choice = data["choices"][0]
        message = choice["message"]
        blocks: list[Block] = []
        if message.get("content"):
            blocks.append(TextBlock(content=message["content"]))
        for call in message.get("tool_calls") or []:
            blocks.append(
                FunctionCallBlock(
                    id=call["id"],
                    name=call["function"]["name"],
                    arguments=_loads(call["function"]["arguments"], call["function"]["name"]),
                )
            )
        return ClientResponse(
            blocks=blocks,
            stop_reason=choice.get("finish_reason"),
            usage=_usage(data.get("usage")),
        )

    def _new_stream(self) -> StreamAccumulator:
        return _OpenAIStream()


class _OpenAIStream(StreamAccumulator):
    def __init__(self) -> None:
        self.text: list[str] = []
        self.calls: dict[int, dict[str, Any]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, int] = {}

    def feed(self, event: str, data: str) -> list[str]:
        if data.strip() == "[DONE]":
            return []
        chunk = json.loads(data)
        if "error" in chunk:
            raise RuntimeError(f"Errore dal provider: {chunk['error']}")
        if chunk.get("usage"):
            self.usage = _usage(chunk["usage"])
        out: list[str] = []
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self.text.append(delta["content"])
                out.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                slot = self.calls.setdefault(call.get("index", 0), {"id": "", "name": "", "args": []})
                if call.get("id"):
                    slot["id"] = call["id"]
                function = call.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                if function.get("arguments"):
                    slot["args"].append(function["arguments"])
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        return out

    def finish(self) -> ClientResponse:
        blocks: list[Block] = []
        if self.text:
            blocks.append(TextBlock(content="".join(self.text)))
        for index in sorted(self.calls):
            slot = self.calls[index]
            blocks.append(
                FunctionCallBlock(
                    id=slot["id"], name=slot["name"], arguments=_loads("".join(slot["args"]), slot["name"])
                )
            )
        return ClientResponse(blocks=blocks, stop_reason=self.finish_reason, usage=self.usage)
