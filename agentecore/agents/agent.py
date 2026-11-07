import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Callable, Iterator

from agentecore.core.clients.client import Client, ClientResponse, StreamEvent, ToolExecution
from agentecore.tools.tool import Tool
from agentecore.type.type import Block, FunctionCallBlock, Memory, Role, TextBlock

logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        client: Client,
        tools: list[Tool] | None = None,
        system_prompt: str = "",
        max_steps: int = 10,
        memory: Memory | None = None,
        on_tool: Callable[[ToolExecution], None] | None = None,
    ):
        self.client = client
        self.tools = {t.name: t for t in tools or []}
        self._tool_specs = [t.to_dict() for t in self.tools.values()]
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.memory = memory or Memory()
        # Chiamata subito dopo ogni tool eseguito, anche con run()/arun().
        self.on_tool = on_tool

    def run(self, input: str | list[Block]) -> ClientResponse:
        self._remember_input(input)
        used: list[ToolExecution] = []

        for step in range(self.max_steps):
            response = self.client.invoke(
                [],
                tools=self._tool_specs,
                memory=self.memory,
                system_prompt=self.system_prompt or None,
            )
            calls = response.function_calls
            if not calls:
                self.memory.add(Role.ASSISTANT, response.blocks)
                response.tool_calls = used
                return response

            logger.debug("step %d: %d tool call", step, len(calls))
            executions = [self._execute(call) for call in calls]
            self._remember_tool_step(response.blocks, executions)
            used.extend(executions)
            self._notify(executions)

        raise RuntimeError(f"L'agente non ha finito entro {self.max_steps} passi")

    def _remember_input(self, input: str | list[Block]) -> None:
        blocks = [TextBlock(content=input)] if isinstance(input, str) else input
        if blocks:
            self.memory.add(Role.USER, blocks)

    def _notify(self, executions: list[ToolExecution]) -> None:
        if self.on_tool is None:
            return
        for execution in executions:
            self.on_tool(execution)

    def _remember_tool_step(self, assistant_blocks: list[Block], executions: list[ToolExecution]) -> None:
        # Le tool call dell'assistente e i loro risultati si salvano insieme, dopo
        # l'esecuzione: così la memoria non contiene mai tool call senza risposta.
        self.memory.add(Role.ASSISTANT, assistant_blocks)
        self.memory.add(Role.TOOL, [e.block for e in executions])

    async def arun(self, input: str | list[Block]) -> ClientResponse:
        self._remember_input(input)
        used: list[ToolExecution] = []

        for step in range(self.max_steps):
            response = await self.client.a_invoke(
                [],
                tools=self._tool_specs,
                memory=self.memory,
                system_prompt=self.system_prompt or None,
            )
            calls = response.function_calls
            if not calls:
                self.memory.add(Role.ASSISTANT, response.blocks)
                response.tool_calls = used
                return response

            logger.debug("step %d: %d tool call", step, len(calls))
            executions = list(await asyncio.gather(*(self._aexecute(c) for c in calls)))
            self._remember_tool_step(response.blocks, executions)
            used.extend(executions)
            self._notify(executions)

        raise RuntimeError(f"L'agente non ha finito entro {self.max_steps} passi")

    def stream(self, input: str | list[Block]) -> Iterator[StreamEvent]:
        """Come run(), ma emette il testo man mano che arriva.

        Eventi: `text` (frammenti), `tool` (un tool è stato eseguito), `done` (risposta finale).
        """
        self._remember_input(input)
        used: list[ToolExecution] = []

        for step in range(self.max_steps):
            response: ClientResponse | None = None
            for event in self.client.stream_invoke(
                [], tools=self._tool_specs, memory=self.memory, system_prompt=self.system_prompt or None
            ):
                if event.type == "text":
                    yield event
                else:
                    response = event.response
            assert response is not None
            calls = response.function_calls
            if not calls:
                self.memory.add(Role.ASSISTANT, response.blocks)
                response.tool_calls = used
                yield StreamEvent("done", response=response)
                return

            logger.debug("step %d: %d tool call", step, len(calls))
            executions = [self._execute(call) for call in calls]
            self._remember_tool_step(response.blocks, executions)
            used.extend(executions)
            self._notify(executions)
            for execution in executions:
                yield StreamEvent("tool", result=execution.block, call=execution)

        raise RuntimeError(f"L'agente non ha finito entro {self.max_steps} passi")

    async def astream(self, input: str | list[Block]) -> AsyncIterator[StreamEvent]:
        self._remember_input(input)
        used: list[ToolExecution] = []

        for step in range(self.max_steps):
            response: ClientResponse | None = None
            async for event in self.client.a_stream_invoke(
                [], tools=self._tool_specs, memory=self.memory, system_prompt=self.system_prompt or None
            ):
                if event.type == "text":
                    yield event
                else:
                    response = event.response
            assert response is not None
            calls = response.function_calls
            if not calls:
                self.memory.add(Role.ASSISTANT, response.blocks)
                response.tool_calls = used
                yield StreamEvent("done", response=response)
                return

            logger.debug("step %d: %d tool call", step, len(calls))
            executions = list(await asyncio.gather(*(self._aexecute(c) for c in calls)))
            self._remember_tool_step(response.blocks, executions)
            used.extend(executions)
            self._notify(executions)
            for execution in executions:
                yield StreamEvent("tool", result=execution.block, call=execution)

        raise RuntimeError(f"L'agente non ha finito entro {self.max_steps} passi")

    async def _aexecute(self, call: FunctionCallBlock) -> ToolExecution:
        start = time.perf_counter()
        tool = self.tools.get(call.name)
        if tool is None:
            return self._execution(call, f"Errore: il tool '{call.name}' non esiste", True, start)
        try:
            return self._execution(call, self._stringify(await tool.arun(call.arguments)), False, start)
        except Exception as e:
            logger.exception("tool %s fallito", call.name)
            return self._execution(call, f"Errore nell'esecuzione di '{call.name}': {e}", True, start)

    def _execute(self, call: FunctionCallBlock) -> ToolExecution:
        start = time.perf_counter()
        tool = self.tools.get(call.name)
        if tool is None:
            return self._execution(call, f"Errore: il tool '{call.name}' non esiste", True, start)
        try:
            return self._execution(call, self._stringify(tool.run(call.arguments)), False, start)
        except Exception as e:
            logger.exception("tool %s fallito", call.name)
            return self._execution(call, f"Errore nell'esecuzione di '{call.name}': {e}", True, start)

    @staticmethod
    def _execution(call: FunctionCallBlock, result: str, error: bool, start: float) -> ToolExecution:
        return ToolExecution(
            id=call.id,
            name=call.name,
            arguments=call.arguments,
            result=result,
            error=error,
            duration=time.perf_counter() - start,
        )

    @staticmethod
    def _stringify(value: Any) -> str:
        return value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
