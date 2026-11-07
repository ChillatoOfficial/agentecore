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


class Anthropic(HTTPClient):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        system_prompt: str = "",
        temperature: float | None = None,
        max_tokens: int = 4096,
        **http_options: Any,
    ):
        super().__init__(model_name, system_prompt, temperature, base_url=base_url, **http_options)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens

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
        messages = [
            self._to_message(turn.role, turn.blocks)
            for turn in memory.turns
            if turn.role != Role.SYSTEM and turn.blocks
        ]
        if input:
            messages.append(self._to_message(Role.USER, input))

        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
        if stream:
            payload["stream"] = True

        return (
            f"{self.base_url}/messages",
            {"x-api-key": str(self.api_key), "anthropic-version": "2023-06-01"},
            payload,
        )

    @staticmethod
    def _media_block(media: Media) -> dict[str, Any]:
        if media.source_type == "url":
            source: dict[str, Any] = {"type": "url", "url": media.source}
        else:
            source = {"type": "base64", "media_type": media.mime_type, "data": media.data_base64()}
        return {"type": "image" if media.is_image else "document", "source": source}

    @classmethod
    def _to_message(cls, role: Role, blocks: list[Block]) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for b in blocks:
            if isinstance(b, TextBlock):
                content.append({"type": "text", "text": b.content})
            elif isinstance(b, MediaBlock) and b.media is not None:
                content.append(cls._media_block(b.media))
            elif isinstance(b, FunctionCallBlock):
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments})
            elif isinstance(b, FunctionCallResultBlock):
                content.append({"type": "tool_result", "tool_use_id": b.id, "content": b.result})
        return {"role": role.anthropic_role, "content": content}

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ClientResponse:
        blocks: list[Block] = []
        for part in data.get("content", []):
            if part["type"] == "text":
                blocks.append(TextBlock(content=part["text"]))
            elif part["type"] == "tool_use":
                blocks.append(FunctionCallBlock(id=part["id"], name=part["name"], arguments=part["input"]))
        usage = data.get("usage") or {}
        return ClientResponse(
            blocks=blocks,
            stop_reason=data.get("stop_reason"),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            },
        )

    def _new_stream(self) -> StreamAccumulator:
        return _AnthropicStream()


class _AnthropicStream(StreamAccumulator):
    def __init__(self) -> None:
        self.parts: dict[int, dict[str, Any]] = {}
        self.stop_reason: str | None = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def feed(self, event: str, data: str) -> list[str]:
        chunk = json.loads(data)
        kind = chunk.get("type", event)
        out: list[str] = []

        if kind == "message_start":
            usage = (chunk.get("message") or {}).get("usage") or {}
            self.usage["prompt_tokens"] = usage.get("input_tokens", 0)
        elif kind == "content_block_start":
            block = chunk["content_block"]
            if block["type"] == "text":
                self.parts[chunk["index"]] = {"type": "text", "text": []}
                if block.get("text"):
                    self.parts[chunk["index"]]["text"].append(block["text"])
                    out.append(block["text"])
            elif block["type"] == "tool_use":
                self.parts[chunk["index"]] = {
                    "type": "tool_use", "id": block["id"], "name": block["name"], "json": [],
                }
        elif kind == "content_block_delta":
            part = self.parts.get(chunk["index"])
            delta = chunk["delta"]
            if part is not None and delta["type"] == "text_delta":
                part["text"].append(delta["text"])
                out.append(delta["text"])
            elif part is not None and delta["type"] == "input_json_delta":
                part["json"].append(delta["partial_json"])
        elif kind == "message_delta":
            self.stop_reason = (chunk.get("delta") or {}).get("stop_reason") or self.stop_reason
            usage = chunk.get("usage") or {}
            if "output_tokens" in usage:
                self.usage["completion_tokens"] = usage["output_tokens"]
        elif kind == "error":
            raise RuntimeError(f"Errore dal provider: {chunk.get('error')}")
        return out

    def finish(self) -> ClientResponse:
        blocks: list[Block] = []
        for index in sorted(self.parts):
            part = self.parts[index]
            if part["type"] == "text":
                text = "".join(part["text"])
                if text:
                    blocks.append(TextBlock(content=text))
            else:
                blocks.append(
                    FunctionCallBlock(
                        id=part["id"], name=part["name"], arguments=_loads("".join(part["json"]), part["name"])
                    )
                )
        return ClientResponse(blocks=blocks, stop_reason=self.stop_reason, usage=self.usage)
