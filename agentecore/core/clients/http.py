import asyncio
import logging
import random
import time
from abc import abstractmethod
from typing import Any, AsyncIterator, Iterator

import httpx

from agentecore.core.clients.client import Client, ClientResponse, StreamEvent
from agentecore.core.clients.stream import SSEParser, StreamAccumulator
from agentecore.type.type import Block, Memory

logger = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
MAX_RETRY_AFTER = 30.0


class HTTPClient(Client):
    """Client per provider con API HTTP/JSON.

    Le sottoclassi costruiscono la richiesta e leggono la risposta; qui ci sono
    pool di connessioni riutilizzate (keep-alive), I/O sincrono e asincrono
    nativo, retry con backoff esponenziale e chiusura delle risorse.
    """

    def __init__(
        self,
        model_name: str,
        system_prompt: str = "",
        temperature: float | None = None,
        *,
        base_url: str,
        timeout: float = 60.0,
        connect_timeout: float = 10.0,
        max_retries: int = 3,
        max_connections: int = 100,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ):
        super().__init__(model_name, system_prompt, temperature)
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )
        self._transport = transport
        self._client: httpx.Client | None = None
        self._a_client: httpx.AsyncClient | None = None
        self._a_loop: asyncio.AbstractEventLoop | None = None

    @abstractmethod
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
        """Restituisce (url, headers, payload JSON); con `stream=True` la richiesta per lo streaming SSE."""

    @abstractmethod
    def _parse_response(self, data: dict[str, Any]) -> ClientResponse:
        """Converte il JSON del provider in un ClientResponse."""

    def _new_stream(self) -> StreamAccumulator | None:
        """Accumulatore degli eventi SSE del provider. None = streaming non supportato."""
        return None

    def _invoke(self, **kwargs: Any) -> ClientResponse:
        url, headers, payload = self._build_request(**kwargs)
        return self._parse_response(self._post(url, headers, payload))

    async def _a_invoke(self, **kwargs: Any) -> ClientResponse:
        url, headers, payload = self._build_request(**kwargs)
        return self._parse_response(await self._a_post(url, headers, payload))

    def _stream_invoke(self, **kwargs: Any) -> Iterator[StreamEvent]:
        accumulator = self._new_stream()
        if accumulator is None:
            yield from super()._stream_invoke(**kwargs)
            return
        url, headers, payload = self._build_request(**kwargs, stream=True)
        for event, data in self._sse(url, headers, payload):
            for text in accumulator.feed(event, data):
                yield StreamEvent("text", text=text)
        yield StreamEvent("done", response=accumulator.finish())

    async def _a_stream_invoke(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:
        accumulator = self._new_stream()
        if accumulator is None:
            async for event in super()._a_stream_invoke(**kwargs):
                yield event
            return
        url, headers, payload = self._build_request(**kwargs, stream=True)
        async for event, data in self._a_sse(url, headers, payload):
            for text in accumulator.feed(event, data):
                yield StreamEvent("text", text=text)
        yield StreamEvent("done", response=accumulator.finish())

    # --- trasporto -------------------------------------------------------

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self._timeout, limits=self._limits, transport=self._transport
            )
        return self._client

    def _get_a_client(self) -> httpx.AsyncClient:
        # Un AsyncClient è legato all'event loop in cui è nato: se il loop
        # cambia (es. due asyncio.run consecutivi) se ne crea uno nuovo.
        loop = asyncio.get_running_loop()
        if self._a_client is None or self._a_loop is not loop:
            self._a_client = httpx.AsyncClient(
                timeout=self._timeout, limits=self._limits, transport=self._transport
            )
            self._a_loop = loop
        return self._a_client

    def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            try:
                response = client.post(url, headers=headers, json=payload)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                delay = self._backoff(attempt, None)
            else:
                delay = self._retry_delay(attempt, response)
                if delay is None:
                    return self._json_or_raise(response)
            logger.warning("richiesta fallita, nuovo tentativo tra %.1fs", delay)
            time.sleep(delay)
        raise AssertionError("irraggiungibile")

    async def _a_post(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_a_client()
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                delay = self._backoff(attempt, None)
            else:
                delay = self._retry_delay(attempt, response)
                if delay is None:
                    return self._json_or_raise(response)
            logger.warning("richiesta fallita, nuovo tentativo tra %.1fs", delay)
            await asyncio.sleep(delay)
        raise AssertionError("irraggiungibile")

    def _sse(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> Iterator[tuple[str, str]]:
        # Retry solo prima del primo byte utile: dopo, ripetere duplicherebbe il testo già emesso.
        client = self._get_client()
        for attempt in range(self.max_retries + 1):
            started = False
            try:
                with client.stream("POST", url, headers=headers, json=payload) as response:
                    delay = self._retry_delay(attempt, response)
                    if delay is None:
                        if response.is_error:
                            response.read()
                            raise self._status_error(response)
                        started = True
                        parser = SSEParser()
                        for line in response.iter_lines():
                            item = parser.feed(line)
                            if item:
                                yield item
                        item = parser.flush()
                        if item:
                            yield item
                        return
            except httpx.TransportError:
                if started or attempt == self.max_retries:
                    raise
                delay = self._backoff(attempt, None)
            logger.warning("richiesta fallita, nuovo tentativo tra %.1fs", delay)
            time.sleep(delay)

    async def _a_sse(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> AsyncIterator[tuple[str, str]]:
        client = self._get_a_client()
        for attempt in range(self.max_retries + 1):
            started = False
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    delay = self._retry_delay(attempt, response)
                    if delay is None:
                        if response.is_error:
                            await response.aread()
                            raise self._status_error(response)
                        started = True
                        parser = SSEParser()
                        async for line in response.aiter_lines():
                            item = parser.feed(line)
                            if item:
                                yield item
                        item = parser.flush()
                        if item:
                            yield item
                        return
            except httpx.TransportError:
                if started or attempt == self.max_retries:
                    raise
                delay = self._backoff(attempt, None)
            logger.warning("richiesta fallita, nuovo tentativo tra %.1fs", delay)
            await asyncio.sleep(delay)

    def _retry_delay(self, attempt: int, response: httpx.Response) -> float | None:
        """None se la risposta è definitiva (successo o errore non ritentabile)."""
        if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
            return self._backoff(attempt, response)
        return None

    @staticmethod
    def _backoff(attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            try:
                return min(float(response.headers["retry-after"]), MAX_RETRY_AFTER)
            except (KeyError, ValueError):
                pass
        return min(0.5 * 2**attempt, 8.0) * (0.5 + random.random())

    @staticmethod
    def _status_error(response: httpx.Response) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError(
            f"{response.status_code} {response.reason_phrase}: {response.text[:500]}",
            request=response.request,
            response=response,
        )

    @classmethod
    def _json_or_raise(cls, response: httpx.Response) -> dict[str, Any]:
        if response.is_error:
            raise cls._status_error(response)
        return response.json()

    # --- ciclo di vita ---------------------------------------------------

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._a_client is not None:
            await self._a_client.aclose()
            self._a_client = None
            self._a_loop = None

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
