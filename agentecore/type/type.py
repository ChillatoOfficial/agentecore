import base64
import logging
import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, Literal
from dataclasses import asdict, dataclass, field
from enum import Enum
from pydantic import BaseModel

logger = logging.getLogger(__name__)
Model = TypeVar("Model", bound=BaseModel)

class Role(Enum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"

    @property
    def google_role(self) -> str:
        if self == Role.ASSISTANT:
            return "model"
        elif self == Role.USER:
            return "user"
        elif self == Role.SYSTEM:
            return "system"
        elif self == Role.TOOL:
            return "user"
        else:
            raise ValueError(f"Unknown role: {self}")

    @property
    def anthropic_role(self) -> str:
        if self == Role.ASSISTANT:
            return "assistant"
        elif self == Role.USER:
            return "user"
        elif self == Role.SYSTEM:
            return "system"
        elif self == Role.TOOL:
            return "user"
        else:
            raise ValueError(f"Unknown role: {self}")


@dataclass
class Block(ABC):
    type: str = field(init=False)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Block":
        for subclass in cls.__subclasses__():
            if subclass.type == data["type"]:
                return subclass(**{k: v for k, v in data.items() if k != "type"})
        raise ValueError(f"Unknown block type: {data['type']}")


@dataclass
class TextBlock(Block):
    content: str = ""
    type: Literal["text"] = field(default="text", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "content": self.content}


@dataclass
class FunctionCallBlock(Block):
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    # Firma di ragionamento restituita da Gemini: va rimandata così com'è nel turno successivo.
    thought_signature: str | None = None
    type: Literal["function_call"] = field(default="function_call", init=False)

    def to_dict(self) -> dict[str, Any]:
        data = {"type": self.type, "id": self.id, "name": self.name, "arguments": self.arguments}
        if self.thought_signature is not None:
            data["thought_signature"] = self.thought_signature
        return data


@dataclass
class FunctionCallResultBlock(Block):
    id: str = ""
    name: str = ""
    result: str = ""
    type: Literal["function_call_result"] = field(default="function_call_result", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "id": self.id, "name": self.name, "result": self.result}


@dataclass
class StructuredBlock(Block):
    content: BaseModel | None = None
    type: Literal["structured"] = field(default="structured", init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "content": self.content.model_dump() if self.content else None}


SUPPORTED_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"}
)


@dataclass
class Media:
    """Immagine o PDF da allegare a un messaggio.

    `source` è un URL, un percorso di file oppure i dati in base64, secondo `source_type`.
    """

    source_type: Literal["url", "path", "base64"]
    source: str
    mime_type: str

    def __post_init__(self) -> None:
        if self.source_type not in ("url", "path", "base64"):
            raise ValueError(f"source_type non valido: {self.source_type}")
        if self.mime_type not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"Tipo non supportato: {self.mime_type}. Supportati: {sorted(SUPPORTED_MIME_TYPES)}"
            )

    @classmethod
    def from_path(cls, path: str | Path, mime_type: str | None = None) -> "Media":
        return cls("path", str(path), mime_type or cls._guess(str(path)))

    @classmethod
    def from_url(cls, url: str, mime_type: str | None = None) -> "Media":
        return cls("url", url, mime_type or cls._guess(url.split("?")[0]))

    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str) -> "Media":
        return cls("base64", base64.b64encode(data).decode("ascii"), mime_type)

    @staticmethod
    def _guess(name: str) -> str:
        mime, _ = mimetypes.guess_type(name)
        if mime is None:
            raise ValueError(f"Impossibile ricavare il tipo di '{name}': passa mime_type")
        return mime

    @property
    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    @property
    def is_pdf(self) -> bool:
        return self.mime_type == "application/pdf"

    def data_base64(self) -> str:
        """Dati in base64. Per gli URL non è disponibile: il file non viene scaricato."""
        if self.source_type == "base64":
            return self.source
        if self.source_type == "path":
            return base64.b64encode(Path(self.source).read_bytes()).decode("ascii")
        raise ValueError("Questo allegato è un URL: usa from_path o from_bytes per inviarlo come dati")


@dataclass
class MediaBlock(Block):
    media: Media | None = None
    type: Literal["media"] = field(default="media", init=False)

    def __post_init__(self) -> None:
        if isinstance(self.media, dict):
            self.media = Media(**self.media)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "media": asdict(self.media) if self.media else None}


@dataclass
class Turn:
    role: Role
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(b.content for b in self.blocks if isinstance(b, TextBlock))

    @property
    def function_calls(self) -> list[FunctionCallBlock]:
        return [b for b in self.blocks if isinstance(b, FunctionCallBlock)]


@dataclass
class Memory:
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: Role, blocks: list[Block]) -> None:
        self.turns.append(Turn(role, blocks))

    def clear(self) -> None:
        self.turns.clear()
