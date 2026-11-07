import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, Field

from agentecore.agents.agent import Agent
from agentecore.clients.anthropic import Anthropic
from agentecore.clients.google import Google
from agentecore.clients.openai import OpenAI
from agentecore.core.clients.client import Client, ClientResponse
from agentecore.core.clients.stream import SSEParser
from agentecore.core.schema import to_gemini_schema
from agentecore.memory.sqlite import SQLiteMemory
from agentecore.tools.tool import tool
from agentecore.type.type import FunctionCallBlock, Media, MediaBlock, Role, TextBlock
from tests.test_smoke import Transport, somma

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 16


def sse(*events, done=False) -> httpx.Response:
    """Risposta SSE. Ogni evento è un dict oppure una tupla (nome_evento, dict)."""
    body = ""
    for e in events:
        name, obj = e if isinstance(e, tuple) else (None, e)
        body += (f"event: {name}\n" if name else "") + f"data: {json.dumps(obj)}\n\n"
    if done:
        body += "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def openai_text(*pieces):
    chunks = [{"choices": [{"delta": {"content": p}}]} for p in pieces]
    chunks[-1]["choices"][0]["finish_reason"] = "stop"
    chunks.append({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    return sse(*chunks, done=True)


def openai_tool_call():
    return sse(
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "somma", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"a": 17'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ', "b": 25}'}}]},
                      "finish_reason": "tool_calls"}]},
        done=True,
    )


def keys(node, found=None):
    found = set() if found is None else found
    if isinstance(node, dict):
        for k, v in node.items():
            found.add(k)
            keys(v, found)
    elif isinstance(node, list):
        for v in node:
            keys(v, found)
    return found


class TestToolSchema(unittest.TestCase):
    def test_docstring_parameters(self):
        @tool
        def meteo(citta: str, giorni: int = 1, unita: Annotated[str, Field(description="C o F")] = "C") -> str:
            """Dà il meteo di una città.

            Args:
                citta: Nome della città.
                giorni (int): Quanti giorni
                    di previsione.
                unita: ignorata: vince Annotated.

            Returns:
                Il testo del meteo.
            """

        schema = meteo.to_dict()
        props = schema["parameters"]["properties"]
        self.assertEqual(schema["description"], "Dà il meteo di una città.")
        self.assertEqual(props["citta"]["description"], "Nome della città.")
        self.assertEqual(props["giorni"]["description"], "Quanti giorni di previsione.")
        self.assertEqual(props["unita"]["description"], "C o F")
        self.assertEqual(schema["parameters"]["required"], ["citta"])

    def test_varargs_are_ignored(self):
        @tool
        def f(a: int, *args, **kwargs):
            """Prova."""

        self.assertEqual(list(f.parameters["properties"]), ["a"])


class Indirizzo(BaseModel):
    via: str
    cap: str | None = None


class TestGeminiSchema(unittest.TestCase):
    def setUp(self):
        @tool
        def spedisci(indirizzo: Indirizzo, priorita: Literal["alta", "bassa"] = "bassa", note: str | None = None):
            """Spedisce un pacco."""

        @tool
        def ping():
            """Senza parametri."""

        self.spedisci, self.ping = spedisci, ping

    def test_converts_to_supported_subset(self):
        schema = to_gemini_schema(self.spedisci.parameters)
        props = schema["properties"]
        self.assertEqual(props["indirizzo"]["type"], "object")
        self.assertTrue(props["indirizzo"]["properties"]["cap"]["nullable"])
        self.assertEqual(props["priorita"]["enum"], ["alta", "bassa"])
        self.assertTrue(props["note"]["nullable"])
        self.assertEqual(props["note"]["type"], "string")
        self.assertEqual(schema["required"], ["indirizzo"])
        self.assertFalse(keys(schema) & {"$ref", "$defs", "default", "additionalProperties", "title", "anyOf"})

    def test_empty_tool_has_no_parameters(self):
        t = self.ping.to_dict()
        d = Google._to_declaration(t)
        self.assertNotIn("parameters", d)

    def test_recursive_schema_is_rejected(self):
        schema = {"$defs": {"A": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}}},
                  "$ref": "#/$defs/A"}
        with self.assertRaises(ValueError):
            to_gemini_schema(schema)


class TestMedia(unittest.TestCase):
    def blocks(self, media):
        return [TextBlock(content="che cos'è?"), MediaBlock(media=media)]

    def test_validation(self):
        with self.assertRaises(ValueError):
            Media("path", "x.exe", "application/x-msdownload")
        with self.assertRaises(ValueError):
            Media.from_url("https://x.it/file")  # tipo non ricavabile
        self.assertTrue(Media.from_url("https://x.it/a.png?v=1").is_image)

    def test_from_path_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.png"
            p.write_bytes(PNG)
            m = Media.from_path(p)
            self.assertEqual(m.mime_type, "image/png")
            self.assertEqual(m.data_base64(), Media.from_bytes(PNG, "image/png").data_base64())

    def test_openai(self):
        t = Transport([{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}])
        OpenAI("m", api_key="k", transport=t).invoke(self.blocks(Media.from_bytes(PNG, "image/png")))
        content = t.calls[0]["json"]["messages"][0]["content"]
        self.assertEqual([p["type"] for p in content], ["text", "image_url"])
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))

        t = Transport([{"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}])
        c = OpenAI("m", api_key="k", transport=t)
        c.invoke(self.blocks(Media.from_url("https://x.it/a.png")))
        self.assertEqual(t.calls[0]["json"]["messages"][0]["content"][1]["image_url"]["url"], "https://x.it/a.png")
        c.invoke(self.blocks(Media.from_bytes(b"%PDF", "application/pdf")))
        self.assertEqual(t.calls[1]["json"]["messages"][0]["content"][1]["type"], "file")
        with self.assertRaises(ValueError):
            c.invoke(self.blocks(Media.from_url("https://x.it/a.pdf")))

    def test_anthropic(self):
        ok = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
        t = Transport([ok])
        c = Anthropic("m", api_key="k", transport=t)
        c.invoke(self.blocks(Media.from_bytes(PNG, "image/png")))
        c.invoke(self.blocks(Media.from_bytes(b"%PDF", "application/pdf")))
        c.invoke(self.blocks(Media.from_url("https://x.it/a.png")))
        parts = [call["json"]["messages"][0]["content"][1] for call in t.calls]
        self.assertEqual(parts[0]["type"], "image")
        self.assertEqual(parts[0]["source"]["media_type"], "image/png")
        self.assertEqual(parts[1]["type"], "document")
        self.assertEqual(parts[2]["source"], {"type": "url", "url": "https://x.it/a.png"})

    def test_google(self):
        t = Transport([{"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}])
        c = Google("m", api_key="k", transport=t)
        c.invoke(self.blocks(Media.from_bytes(PNG, "image/png")))
        part = t.calls[0]["json"]["contents"][0]["parts"][1]
        self.assertEqual(part["inlineData"]["mimeType"], "image/png")
        with self.assertRaises(ValueError):
            c.invoke(self.blocks(Media.from_url("https://x.it/a.png")))

    def test_media_persists_in_sqlite(self):
        with tempfile.TemporaryDirectory() as d:
            with SQLiteMemory(Path(d) / "m.db") as m:
                m.add(Role.USER, self.blocks(Media.from_bytes(PNG, "image/png")))
            with SQLiteMemory(Path(d) / "m.db") as m:
                block = m.turns[0].blocks[1]
                self.assertIsInstance(block, MediaBlock)
                self.assertEqual(block.media.mime_type, "image/png")
                self.assertEqual(block.media.data_base64(), Media.from_bytes(PNG, "image/png").data_base64())


class TestSSEParser(unittest.TestCase):
    def test_events(self):
        p = SSEParser()
        out = [p.feed(line) for line in [": commento", "event: x", "data: a", "data: b", "", "data:c", ""]]
        self.assertEqual([o for o in out if o], [("x", "a\nb"), ("message", "c")])

    def test_flush_without_trailing_blank_line(self):
        p = SSEParser()
        p.feed("data: ultimo")
        self.assertEqual(p.flush(), ("message", "ultimo"))
        self.assertIsNone(p.flush())


def collect(events):
    texts, done = [], None
    for e in events:
        if e.type == "text":
            texts.append(e.text)
        elif e.type == "done":
            done = e.response
    return texts, done


async def acollect(events):
    return collect([e async for e in events])


class TestStreaming(unittest.TestCase):
    def test_openai_text(self):
        t = Transport([openai_text("Fa ", "42")])
        texts, done = collect(OpenAI("m", api_key="k", transport=t).stream_invoke("ciao"))
        self.assertEqual(texts, ["Fa ", "42"])
        self.assertEqual(done.text, "Fa 42")
        self.assertEqual(done.stop_reason, "stop")
        self.assertEqual(done.usage, {"prompt_tokens": 3, "completion_tokens": 2})
        self.assertTrue(t.calls[0]["json"]["stream"])

    def test_openai_tool_call(self):
        _, done = collect(OpenAI("m", api_key="k", transport=Transport([openai_tool_call()])).stream_invoke("x"))
        call = done.function_calls[0]
        self.assertEqual((call.id, call.name, call.arguments), ("c1", "somma", {"a": 17, "b": 25}))

    def test_anthropic(self):
        body = sse(
            ("message_start", {"type": "message_start", "message": {"usage": {"input_tokens": 7}}}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Vedo "}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "un gatto"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("content_block_start", {"type": "content_block_start", "index": 1,
                                     "content_block": {"type": "tool_use", "id": "t1", "name": "somma", "input": {}}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"a": 1,'}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ' "b": 2}'}}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
            ("message_stop", {"type": "message_stop"}),
        )
        t = Transport([body])
        texts, done = collect(Anthropic("m", api_key="k", transport=t).stream_invoke("ciao"))
        self.assertEqual(texts, ["Vedo ", "un gatto"])
        self.assertEqual(done.text, "Vedo un gatto")
        self.assertEqual(done.function_calls[0].arguments, {"a": 1, "b": 2})
        self.assertEqual(done.stop_reason, "tool_use")
        self.assertEqual(done.usage, {"prompt_tokens": 7, "completion_tokens": 9})
        self.assertTrue(t.calls[0]["json"]["stream"])

    def test_google_and_thought_signature(self):
        body = sse(
            {"candidates": [{"content": {"parts": [{"text": "Ora "}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "calcolo"},
                                                   {"functionCall": {"name": "somma", "args": {"a": 1, "b": 2}},
                                                    "thoughtSignature": "SIG"}]}, "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 5}},
        )
        t = Transport([body, {"candidates": [{"content": {"parts": [{"text": "3"}]}}]}])
        c = Google("m", api_key="k", transport=t)
        texts, done = collect(c.stream_invoke("somma"))
        self.assertEqual(texts, ["Ora ", "calcolo"])
        self.assertEqual(done.text, "Ora calcolo")
        self.assertTrue(t.calls[0]["url"].endswith(":streamGenerateContent?alt=sse"))
        self.assertEqual(done.usage, {"prompt_tokens": 4, "completion_tokens": 5})
        call = done.function_calls[0]
        self.assertEqual(call.thought_signature, "SIG")

        # la firma deve tornare indietro nel turno successivo
        agent = Agent(c, tools=[somma])
        agent.memory.add(Role.USER, [TextBlock(content="somma")])
        agent.memory.add(Role.ASSISTANT, [call])
        c.invoke([], memory=agent.memory)
        sent = t.calls[1]["json"]["contents"][1]["parts"][0]
        self.assertEqual(sent["thoughtSignature"], "SIG")

    def test_retry_before_first_byte_and_error(self):
        t = Transport([httpx.Response(503, headers={"retry-after": "0"}), openai_text("ok")])
        texts, _ = collect(OpenAI("m", api_key="k", transport=t).stream_invoke("x"))
        self.assertEqual(texts, ["ok"])
        self.assertEqual(len(t.calls), 2)

        t = Transport([httpx.Response(400, text="richiesta sbagliata")])
        with self.assertRaises(httpx.HTTPStatusError) as ctx:
            collect(OpenAI("m", api_key="k", transport=t).stream_invoke("x"))
        self.assertIn("richiesta sbagliata", str(ctx.exception))
        self.assertEqual(len(t.calls), 1)

    def test_fallback_for_clients_without_streaming(self):
        class Plain(Client):
            def _invoke(self, **kw):
                return ClientResponse(blocks=[TextBlock(content="intero")])

        texts, done = collect(Plain("m").stream_invoke("x"))
        self.assertEqual((texts, done.text), (["intero"], "intero"))

    def test_agent_stream_with_tool(self):
        t = Transport([openai_tool_call(), openai_text("Fa ", "42")])
        agent = Agent(OpenAI("m", api_key="k", transport=t), tools=[somma])
        events = list(agent.stream("17+25?"))
        self.assertEqual([e.type for e in events], ["tool", "text", "text", "done"])
        self.assertEqual(events[0].result.result, "42")
        self.assertEqual((events[0].call.name, events[0].call.arguments), ("somma", {"a": 17, "b": 25}))
        self.assertFalse(events[0].call.error)
        self.assertEqual([c.name for c in events[-1].response.tool_calls], ["somma"])
        self.assertEqual(events[-1].response.text, "Fa 42")
        self.assertEqual([x.role for x in agent.memory.turns],
                         [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT])


class TestAsyncStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_a_stream_invoke(self):
        t = Transport([openai_text("Fa ", "42")])
        async with OpenAI("m", api_key="k", transport=t) as c:
            texts, done = await acollect(c.a_stream_invoke("ciao"))
        self.assertEqual((texts, done.text), (["Fa ", "42"], "Fa 42"))

    async def test_retry_and_error(self):
        t = Transport([httpx.Response(529, headers={"retry-after": "0"}), openai_text("ok")])
        async with OpenAI("m", api_key="k", transport=t) as c:
            texts, _ = await acollect(c.a_stream_invoke("x"))
        self.assertEqual(texts, ["ok"])
        t = Transport([httpx.Response(401, text="chiave errata")])
        async with OpenAI("m", api_key="k", transport=t) as c:
            with self.assertRaises(httpx.HTTPStatusError):
                await acollect(c.a_stream_invoke("x"))

    async def test_agent_astream_with_tool(self):
        t = Transport([openai_tool_call(), openai_text("Fa 42")])
        async with OpenAI("m", api_key="k", transport=t) as c:
            events = [e async for e in Agent(c, tools=[somma]).astream("17+25?")]
        self.assertEqual([e.type for e in events], ["tool", "text", "done"])
        self.assertEqual(events[0].call.result, "42")

    async def test_early_break_does_not_leak(self):
        t = Transport([openai_text("a", "b", "c")])
        async with OpenAI("m", api_key="k", transport=t) as c:
            gen = c.a_stream_invoke("x")
            async for e in gen:
                break
            await gen.aclose()


if __name__ == "__main__":
    unittest.main()
