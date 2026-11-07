import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from agentecore.agents.agent import Agent
from agentecore.clients.anthropic import Anthropic
from agentecore.clients.google import Google
from agentecore.clients.openai import OpenAI
from agentecore.memory.sqlite import SQLiteMemory
from agentecore.tools.tool import tool
from agentecore.type.type import Block, FunctionCallBlock, Role, TextBlock


@tool
def somma(a: int, b: int) -> int:
    """Somma due numeri."""
    return a + b


@tool(name="somma")
async def somma_async(a: int, b: int) -> int:
    """Somma due numeri (async)."""
    return a + b


class Transport(httpx.MockTransport):
    """MockTransport che risponde in sequenza e registra le richieste.

    Ogni elemento di `responses` è un dict (200 con quel JSON) oppure una
    httpx.Response / un'eccezione da restituire o sollevare.
    """

    def __init__(self, responses):
        self.calls = []
        self._responses = list(responses)
        super().__init__(self._handle)

    def _handle(self, request):
        self.calls.append({
            "url": str(request.url),
            "headers": dict(request.headers),
            "json": json.loads(request.content),
        })
        item = self._responses[min(len(self.calls), len(self._responses)) - 1]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, httpx.Response):
            return item
        return httpx.Response(200, json=item)


OPENAI_RESPONSES = [
    {
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "somma", "arguments": '{"a": 17, "b": 25}'}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    },
    {"choices": [{"finish_reason": "stop", "message": {"content": "Fa 42"}}], "usage": {}},
]

ANTHROPIC_RESPONSES = [
    {"content": [{"type": "tool_use", "id": "t1", "name": "somma", "input": {"a": 17, "b": 25}}],
     "stop_reason": "tool_use", "usage": {"input_tokens": 5, "output_tokens": 2}},
    {"content": [{"type": "text", "text": "Fa 42"}], "stop_reason": "end_turn", "usage": {}},
]

GOOGLE_RESPONSES = [
    {"candidates": [{"finishReason": "STOP", "content": {"parts": [
        {"functionCall": {"name": "somma", "args": {"a": 17, "b": 25}}}]}}]},
    {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "Fa 42"}]}}]},
]

OK_OPENAI = OPENAI_RESPONSES[1]


class TestTool(unittest.TestCase):
    def test_schema(self):
        d = somma.to_dict()
        self.assertEqual(d["name"], "somma")
        self.assertEqual(d["description"], "Somma due numeri.")
        self.assertEqual(d["parameters"]["required"], ["a", "b"])
        self.assertNotIn("title", d["parameters"])

    def test_schema_is_cached(self):
        self.assertIs(somma.parameters, somma.parameters)

    def test_run_validates(self):
        self.assertEqual(somma.run({"a": 1, "b": 2}), 3)
        with self.assertRaises(Exception):
            somma.run({"a": "x", "b": 2})


class TestBlocks(unittest.TestCase):
    def test_roundtrip(self):
        b = Block.from_dict(FunctionCallBlock(id="1", name="n", arguments={"x": 1}).to_dict())
        self.assertIsInstance(b, FunctionCallBlock)
        self.assertEqual(b.arguments, {"x": 1})

    def test_roles(self):
        self.assertEqual(Role.ASSISTANT.google_role, "model")
        self.assertEqual(Role.TOOL.google_role, "user")
        self.assertEqual(Role.TOOL.anthropic_role, "user")


class TestProviders(unittest.TestCase):
    def run_agent(self, client, transport):
        with client:
            agent = Agent(client, tools=[somma], system_prompt="sys")
            result = agent.run("quanto fa 17+25?")
        self.assertEqual(result.text, "Fa 42")
        self.assertEqual(len(transport.calls), 2)
        return agent

    def test_openai(self):
        t = Transport(OPENAI_RESPONSES)
        self.run_agent(OpenAI("m", api_key="k", transport=t), t)
        msgs = t.calls[1]["json"]["messages"]
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "tool"])
        self.assertEqual(msgs[3]["tool_call_id"], "c1")
        self.assertEqual(msgs[3]["content"], "42")
        self.assertEqual(t.calls[0]["headers"]["authorization"], "Bearer k")
        self.assertTrue(t.calls[0]["url"].endswith("/chat/completions"))

    def test_anthropic(self):
        t = Transport(ANTHROPIC_RESPONSES)
        self.run_agent(Anthropic("m", api_key="k", transport=t), t)
        payload = t.calls[1]["json"]
        self.assertEqual(payload["system"], "sys")
        self.assertEqual([m["role"] for m in payload["messages"]], ["user", "assistant", "user"])
        self.assertEqual(payload["messages"][2]["content"][0]["type"], "tool_result")
        self.assertIn("input_schema", payload["tools"][0])
        self.assertEqual(t.calls[0]["headers"]["x-api-key"], "k")

    def test_google(self):
        t = Transport(GOOGLE_RESPONSES)
        self.run_agent(Google("m", api_key="k", transport=t), t)
        payload = t.calls[1]["json"]
        self.assertEqual([c["role"] for c in payload["contents"]], ["user", "model", "user"])
        self.assertIn("functionResponse", payload["contents"][2]["parts"][0])
        self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "sys")
        self.assertEqual(t.calls[0]["headers"]["x-goog-api-key"], "k")


class TestAgent(unittest.TestCase):
    def test_unknown_tool_and_max_steps(self):
        looping = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "boh", "arguments": "{}"}}]}}]}
        agent = Agent(OpenAI("m", api_key="k", transport=Transport([looping])), tools=[somma], max_steps=3)
        with self.assertRaises(RuntimeError):
            agent.run("ciao")
        self.assertIn("non esiste", agent.memory.turns[2].blocks[0].result)

    def test_tool_calls_visible_in_run_and_callback(self):
        seen = []
        call_somma = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "somma", "arguments": '{"a": 17, "b": 25}'}}]}}]}
        t = Transport([call_somma, {"choices": [{"finish_reason": "stop", "message": {"content": "42"}}]}])
        agent = Agent(OpenAI("m", api_key="k", transport=t), tools=[somma], on_tool=seen.append)
        response = agent.run("17+25?")
        self.assertEqual(len(response.tool_calls), 1)
        call = response.tool_calls[0]
        self.assertEqual((call.name, call.arguments, call.result, call.error), ("somma", {"a": 17, "b": 25}, "42", False))
        self.assertGreaterEqual(call.duration, 0)
        self.assertEqual(seen, response.tool_calls)

    def test_failed_tool_is_flagged(self):
        bad = {"choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "boh", "arguments": "{}"}}]}}]}
        done = {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        agent = Agent(OpenAI("m", api_key="k", transport=Transport([bad, done])), tools=[somma])
        call = agent.run("x").tool_calls[0]
        self.assertTrue(call.error)
        self.assertIn("non esiste", call.result)


class TestHTTP(unittest.TestCase):
    def client(self, responses, **kw):
        t = Transport(responses)
        return OpenAI("m", api_key="k", transport=t, **kw), t

    def test_retry_on_429_then_ok(self):
        c, t = self.client([httpx.Response(429, headers={"retry-after": "0"}), OK_OPENAI])
        self.assertEqual(c.invoke("ciao").stop_reason, "stop")
        self.assertEqual(len(t.calls), 2)

    def test_retry_on_transport_error(self):
        c, t = self.client([httpx.ConnectError("boom"), OK_OPENAI], max_retries=2)
        c._backoff = lambda attempt, response: 0
        self.assertEqual(c.invoke("ciao").stop_reason, "stop")
        self.assertEqual(len(t.calls), 2)

    def test_gives_up_after_max_retries(self):
        c, t = self.client([httpx.Response(503, headers={"retry-after": "0"})], max_retries=2)
        with self.assertRaises(httpx.HTTPStatusError):
            c.invoke("ciao")
        self.assertEqual(len(t.calls), 3)

    def test_no_retry_on_400_and_error_body(self):
        c, t = self.client([httpx.Response(400, text="modello sconosciuto")])
        with self.assertRaises(httpx.HTTPStatusError) as ctx:
            c.invoke("ciao")
        self.assertIn("modello sconosciuto", str(ctx.exception))
        self.assertEqual(len(t.calls), 1)

    def test_connection_pool_reused_and_closed(self):
        c, _ = self.client([OK_OPENAI])
        c.invoke("a")
        first = c._client
        c.invoke("b")
        self.assertIs(c._client, first)
        c.close()
        self.assertIsNone(c._client)


class TestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_arun_sync_and_async_tools(self):
        for t in (somma, somma_async):
            self.assertEqual(await t.arun({"a": 1, "b": 2}), 3)

    def test_run_rejects_async_tool(self):
        with self.assertRaises(TypeError):
            somma_async.run({"a": 1, "b": 2})

    async def test_agent_arun_openai(self):
        t = Transport(OPENAI_RESPONSES)
        async with OpenAI("m", api_key="k", transport=t) as client:
            agent = Agent(client, tools=[somma_async], system_prompt="sys")
            result = await agent.arun("quanto fa 17+25?")
        self.assertEqual(result.text, "Fa 42")
        self.assertEqual(t.calls[1]["json"]["messages"][3]["content"], "42")

    async def test_a_invoke_native_and_retry(self):
        t = Transport([httpx.Response(529, headers={"retry-after": "0"}), ANTHROPIC_RESPONSES[1]])
        async with Anthropic("m", api_key="k", transport=t) as c:
            r = await c.a_invoke("ciao")
        self.assertEqual(r.text, "Fa 42")
        self.assertEqual(len(t.calls), 2)

    async def test_parallel_tool_calls(self):
        @tool(name="lento")
        async def lento() -> str:
            """Attende."""
            await asyncio.sleep(0.2)
            return "ok"

        def call(i):
            return {"id": f"c{i}", "type": "function", "function": {"name": "lento", "arguments": "{}"}}

        responses = [
            {"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
                                                                      "tool_calls": [call(i) for i in range(5)]}}]},
            OK_OPENAI,
        ]
        async with OpenAI("m", api_key="k", transport=Transport(responses)) as client:
            start = time.perf_counter()
            await Agent(client, tools=[lento]).arun("vai")
            elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.6)  # 5 x 0.2s in sequenza sarebbero 1.0s

    def test_client_survives_event_loop_change(self):
        c = OpenAI("m", api_key="k", transport=Transport([OK_OPENAI]))
        self.assertEqual(asyncio.run(c.a_invoke("a")).stop_reason, "stop")
        self.assertEqual(asyncio.run(c.a_invoke("b")).stop_reason, "stop")


class TestSQLiteMemory(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "memory.db"

    def tearDown(self):
        self.dir.cleanup()

    def test_persists_across_instances(self):
        with SQLiteMemory(self.path, "anna") as m:
            m.add(Role.USER, [TextBlock(content="ciao è")])
            m.add(Role.ASSISTANT, [FunctionCallBlock(id="1", name="somma", arguments={"a": 1})])
        with SQLiteMemory(self.path, "anna") as m:
            self.assertEqual([t.role for t in m.turns], [Role.USER, Role.ASSISTANT])
            self.assertEqual(m.turns[0].text, "ciao è")
            self.assertEqual(m.turns[1].function_calls[0].arguments, {"a": 1})

    def test_sessions_are_isolated(self):
        with SQLiteMemory(self.path, "anna") as a, SQLiteMemory(self.path, "luca") as b:
            a.add(Role.USER, [TextBlock(content="a")])
            b.add(Role.USER, [TextBlock(content="b")])
            self.assertEqual({s["session_id"]: s["turns"] for s in a.list_sessions()}, {"anna": 1, "luca": 1})
            a.delete_session("luca")
            self.assertEqual([s["session_id"] for s in a.list_sessions()], ["anna"])
            a.clear()
            self.assertEqual(a.list_sessions(), [])
        with SQLiteMemory(self.path, "anna") as a:
            self.assertEqual(a.turns, [])

    def test_agent_resumes_conversation(self):
        with SQLiteMemory(self.path, "anna") as m:
            Agent(OpenAI("m", api_key="k", transport=Transport([OK_OPENAI])), memory=m).run("primo")
        t = Transport([OK_OPENAI])
        with SQLiteMemory(self.path, "anna") as m:
            Agent(OpenAI("m", api_key="k", transport=t), memory=m).run("secondo")
        roles = [x["role"] for x in t.calls[0]["json"]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_failed_call_leaves_no_dangling_tool_call(self):
        t = Transport([OPENAI_RESPONSES[0], httpx.Response(400, text="no")])
        with SQLiteMemory(self.path, "anna") as m:
            agent = Agent(OpenAI("m", api_key="k", transport=t), tools=[somma], memory=m)
            with self.assertRaises(httpx.HTTPStatusError):
                agent.run("17+25?")
        with SQLiteMemory(self.path, "anna") as m:
            self.assertEqual([x.role for x in m.turns], [Role.USER, Role.ASSISTANT, Role.TOOL])
            self.assertEqual(m.turns[2].blocks[0].result, "42")


if __name__ == "__main__":
    unittest.main()
