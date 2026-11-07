import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Literal

from agentecore.type.type import Block, FunctionCallBlock, FunctionCallResultBlock, Memory, TextBlock

logger = logging.getLogger(__name__)


@dataclass
class ToolExecution:
    """Un tool eseguito dall'agente: cosa è stato chiamato, con quali argomenti e con che esito."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    error: bool = False
    duration: float = 0.0  # secondi

    @property
    def block(self) -> FunctionCallResultBlock:
        return FunctionCallResultBlock(id=self.id, name=self.name, result=self.result)


@dataclass
class ClientResponse:
    blocks: list[Block] = field(default_factory=list)
    stop_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    # Compilato dall'Agent: tutti i tool eseguiti per arrivare a questa risposta, in ordine.
    tool_calls: list[ToolExecution] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(b.content for b in self.blocks if isinstance(b, TextBlock))

    @property
    def function_calls(self) -> list[FunctionCallBlock]:
        return [b for b in self.blocks if isinstance(b, FunctionCallBlock)]


@dataclass
class StreamEvent:
    """Evento di uno stream.

    - `text`: un frammento di testo (`text`)
    - `tool`: un tool è stato eseguito dall'agente (`call` con argomenti, esito e durata; `result` è il solo risultato)
    - `done`: la risposta è completa (`response`, con testo intero e tool call)
    """

    type: Literal["text", "tool", "done"]
    text: str = ""
    result: FunctionCallResultBlock | None = None
    call: ToolExecution | None = None
    response: ClientResponse | None = None


class Client(ABC):
    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        temperature: float | None = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature

    def _prepare(
        self,
        input: str | list[Block],
        tools: list[Any] | None,
        memory: Memory | None,
        temperature: float | None,
        system_prompt: str | None,
    ) -> dict[str, Any]:
        return {
            "input": self._normalize_input(input),
            "tools": tools or [],
            "memory": memory or Memory(),
            "temperature": self.temperature if temperature is None else temperature,
            "system_prompt": self.system_prompt if system_prompt is None else system_prompt,
        }

    def invoke(
        self,
        input: str | list[Block],
        *,
        tools: list[Any] | None = None,
        memory: Memory | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> ClientResponse:
        return self._invoke(**self._prepare(input, tools, memory, temperature, system_prompt))

    async def a_invoke(
        self,
        input: str | list[Block],
        *,
        tools: list[Any] | None = None,
        memory: Memory | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> ClientResponse:
        return await self._a_invoke(**self._prepare(input, tools, memory, temperature, system_prompt))

    def stream_invoke(
        self,
        input: str | list[Block],
        *,
        tools: list[Any] | None = None,
        memory: Memory | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Risposta a frammenti: eventi `text`, poi un evento `done` con la risposta completa."""
        return self._stream_invoke(**self._prepare(input, tools, memory, temperature, system_prompt))

    def a_stream_invoke(
        self,
        input: str | list[Block],
        *,
        tools: list[Any] | None = None,
        memory: Memory | None = None,
        temperature: float | None = None,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._a_stream_invoke(**self._prepare(input, tools, memory, temperature, system_prompt))

    async def _a_invoke(self, **kwargs: Any) -> ClientResponse:
        # Default: esegue la versione sincrona in un thread. Un provider può
        # sovrascriverlo con un client HTTP davvero asincrono.
        return await asyncio.to_thread(self._invoke, **kwargs)

    def _stream_invoke(self, **kwargs: Any) -> Iterator[StreamEvent]:
        # Default per i provider senza streaming: una sola risposta intera.
        response = self._invoke(**kwargs)
        if response.text:
            yield StreamEvent("text", text=response.text)
        yield StreamEvent("done", response=response)

    async def _a_stream_invoke(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        response = await self._a_invoke(**kwargs)
        if response.text:
            yield StreamEvent("text", text=response.text)
        yield StreamEvent("done", response=response)

    @staticmethod
    def _normalize_input(input: str | list[Block]) -> list[Block]:
        if isinstance(input, str):
            return [TextBlock(content=input)] if input else []
        return input

    @abstractmethod
    def _invoke(
        self,
        *,
        input: list[Block],
        tools: list[Any],
        memory: Memory,
        temperature: float | None,
        system_prompt: str,
    ) -> ClientResponse:
        """Chiama il provider e restituisce la risposta come blocchi."""
