from abc import ABC, abstractmethod

from agentecore.core.clients.client import ClientResponse


class SSEParser:
    """Parser di Server-Sent Events: riceve una riga alla volta, restituisce (evento, dati) a evento completo."""

    def __init__(self) -> None:
        self._event = "message"
        self._data: list[str] = []

    def feed(self, line: str) -> tuple[str, str] | None:
        if line == "":
            return self.flush()
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._event = value
        elif field == "data":
            self._data.append(value)
        return None

    def flush(self) -> tuple[str, str] | None:
        item = (self._event, "\n".join(self._data)) if self._data else None
        self._event = "message"
        self._data = []
        return item


class StreamAccumulator(ABC):
    """Costruisce la risposta completa a partire dagli eventi di un provider."""

    @abstractmethod
    def feed(self, event: str, data: str) -> list[str]:
        """Elabora un evento e restituisce i frammenti di testo che contiene."""

    @abstractmethod
    def finish(self) -> ClientResponse:
        """La risposta completa, da chiamare a stream terminato."""
