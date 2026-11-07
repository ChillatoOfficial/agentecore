import json
import os
import uuid
from typing import Any

from agentecore.core.clients.client import ClientResponse
from agentecore.core.clients.http import HTTPClient
from agentecore.core.clients.stream import StreamAccumulator
from agentecore.core.schema import to_gemini_schema
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


def _usage(data: dict[str, Any] | None) -> dict[str, int]:
    data = data or {}
    return {
        "prompt_tokens": data.get("promptTokenCount", 0),
        "completion_tokens": data.get("candidatesTokenCount", 0),
    }


def _parts_to_blocks(parts: list[dict[str, Any]], blocks: list[Block]) -> list[str]:
    """Aggiunge le parti di una risposta ai blocchi (unendo il testo contiguo) e restituisce i frammenti di testo."""
    texts: list[str] = []
    for part in parts:
        if part.get("thought"):
            continue
        if "text" in part:
            if blocks and isinstance(blocks[-1], TextBlock):
                blocks[-1].content += part["text"]
            else:
                blocks.append(TextBlock(content=part["text"]))
            texts.append(part["text"])
        elif "functionCall" in part:
            call = part["functionCall"]
            blocks.append(
                FunctionCallBlock(
                    id=call.get("id") or uuid.uuid4().hex,
                    name=call["name"],
                    arguments=call.get("args") or {},
                    thought_signature=part.get("thoughtSignature"),
                )
            )
    return texts


class Google(HTTPClient):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        system_prompt: str = "",
        temperature: float | None = None,
        **http_options: Any,
    ):
        super().__init__(model_name, system_prompt, temperature, base_url=base_url, **http_options)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")

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
        contents = [
            self._to_content(turn.role, turn.blocks)
            for turn in memory.turns
            if turn.role != Role.SYSTEM and turn.blocks
        ]
        if input:
            contents.append(self._to_content(Role.USER, input))

        payload: dict[str, Any] = {"contents": contents}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if temperature is not None:
            payload["generationConfig"] = {"temperature": temperature}
        if tools:
            payload["tools"] = [{"functionDeclarations": [self._to_declaration(t) for t in tools]}]

        method = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return (
            f"{self.base_url}/models/{self.model_name}:{method}",
            {"x-goog-api-key": str(self.api_key)},
            payload,
        )

    @staticmethod
    def _to_declaration(tool: dict[str, Any]) -> dict[str, Any]:
        declaration = {"name": tool["name"], "description": tool["description"]}
        parameters = to_gemini_schema(tool["parameters"])
        if parameters.get("properties"):  # Gemini rifiuta oggetti senza proprietà
            declaration["parameters"] = parameters
        return declaration

    @staticmethod
    def _media_part(media: Media) -> dict[str, Any]:
        if media.source_type == "url":
            raise ValueError("Gemini accetta gli allegati solo come dati: usa Media.from_path o Media.from_bytes")
        return {"inlineData": {"mimeType": media.mime_type, "data": media.data_base64()}}

    @classmethod
    def _to_content(cls, role: Role, blocks: list[Block]) -> dict[str, Any]:
        parts: list[dict[str, Any]] = []
        for b in blocks:
            if isinstance(b, TextBlock):
                parts.append({"text": b.content})
            elif isinstance(b, MediaBlock) and b.media is not None:
                parts.append(cls._media_part(b.media))
            elif isinstance(b, FunctionCallBlock):
                part: dict[str, Any] = {"functionCall": {"name": b.name, "args": b.arguments}}
                if b.thought_signature is not None:
                    part["thoughtSignature"] = b.thought_signature
                parts.append(part)
            elif isinstance(b, FunctionCallResultBlock):
                parts.append({"functionResponse": {"name": b.name, "response": {"result": b.result}}})
        return {"role": role.google_role, "parts": parts}

    @staticmethod
    def _parse_response(data: dict[str, Any]) -> ClientResponse:
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini non ha restituito risposte: {data.get('promptFeedback') or data}")
        candidate = candidates[0]
        blocks: list[Block] = []
        _parts_to_blocks(candidate.get("content", {}).get("parts", []), blocks)
        return ClientResponse(
            blocks=blocks,
            stop_reason=candidate.get("finishReason"),
            usage=_usage(data.get("usageMetadata")),
        )

    def _new_stream(self) -> StreamAccumulator:
        return _GoogleStream()


class _GoogleStream(StreamAccumulator):
    def __init__(self) -> None:
        self.blocks: list[Block] = []
        self.finish_reason: str | None = None
        self.usage: dict[str, int] = {}

    def feed(self, event: str, data: str) -> list[str]:
        chunk = json.loads(data)
        if "error" in chunk:
            raise RuntimeError(f"Errore dal provider: {chunk['error']}")
        if chunk.get("usageMetadata"):
            self.usage = _usage(chunk["usageMetadata"])
        candidates = chunk.get("candidates") or []
        if not candidates:
            return []
        candidate = candidates[0]
        self.finish_reason = candidate.get("finishReason") or self.finish_reason
        return _parts_to_blocks(candidate.get("content", {}).get("parts", []), self.blocks)

    def finish(self) -> ClientResponse:
        return ClientResponse(blocks=self.blocks, stop_reason=self.finish_reason, usage=self.usage)
