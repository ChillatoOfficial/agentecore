import asyncio
import inspect
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any, Callable, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo


def _strip_titles(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _strip_titles(v) for k, v in schema.items() if k != "title"}
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    return schema


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]

    @cached_property
    def parameters(self) -> dict[str, Any]:
        return _strip_titles(self.args_model.model_json_schema())

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def _validated_kwargs(self, arguments: dict[str, Any]) -> dict[str, Any]:
        validated = self.args_model(**arguments)
        return {k: getattr(validated, k) for k in self.args_model.model_fields}

    def run(self, arguments: dict[str, Any]) -> Any:
        """Valida gli argomenti arrivati dal modello e chiama la funzione."""
        if inspect.iscoroutinefunction(self.fn):
            raise TypeError(f"Il tool '{self.name}' è asincrono: usa arun()")
        return self.fn(**self._validated_kwargs(arguments))

    async def arun(self, arguments: dict[str, Any]) -> Any:
        """Come run(), ma i tool sincroni girano in un thread per non bloccare il loop."""
        kwargs = self._validated_kwargs(arguments)
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return await asyncio.to_thread(self.fn, **kwargs)


_ARGS_HEADERS = ("args", "arguments", "parameters", "params", "parametri", "argomenti")
_OTHER_HEADERS = (
    "returns", "return", "raises", "yields", "examples", "example", "note", "notes",
    "ritorna", "restituisce", "solleva", "esempi", "esempio",
)
_HEADER_RE = re.compile(r"^(\w+):\s*$")
_PARAM_RE = re.compile(r"^(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Separa un docstring in stile Google in descrizione e descrizioni dei parametri.

    La descrizione è il testo prima della prima sezione (Args, Returns, ...).
    """
    summary: list[str] = []
    params: dict[str, str] = {}
    section = None
    current: str | None = None
    param_indent = 0
    for line in doc.splitlines():
        stripped = line.strip()
        header = _HEADER_RE.match(line) if line == line.lstrip() else None
        if header and header.group(1).lower() in _ARGS_HEADERS + _OTHER_HEADERS:
            section = "args" if header.group(1).lower() in _ARGS_HEADERS else "other"
            current = None
            continue
        if section is None:
            summary.append(stripped)
        elif section == "args" and stripped:
            match = _PARAM_RE.match(stripped)
            indent = len(line) - len(line.lstrip())
            if match and (current is None or indent <= param_indent):
                current = match.group(1).lstrip("*")
                param_indent = indent
                params[current] = match.group(2).strip()
            elif current is not None:
                params[current] = f"{params[current]} {stripped}".strip()
    return "\n".join(summary).strip(), params


def _has_description(annotation: Any) -> bool:
    return get_origin(annotation) is Annotated and any(
        isinstance(m, FieldInfo) and m.description for m in get_args(annotation)[1:]
    )


def _build_args_model(fn: Callable[..., Any], param_docs: dict[str, str] | None = None) -> type[BaseModel]:
    param_docs = param_docs or {}
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(param.name, Any)
        default = ... if param.default is inspect.Parameter.empty else param.default
        doc = param_docs.get(param.name)
        if doc and not _has_description(annotation):
            default = Field(default, description=doc)
        fields[param.name] = (annotation, default)
    return create_model(f"{fn.__name__}_args", **fields)


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Trasforma una funzione in un Tool. Usabile come @tool oppure @tool(name=...)."""

    def wrap(f: Callable[..., Any]) -> Tool:
        doc_description, param_docs = _parse_docstring(inspect.getdoc(f) or "")
        return Tool(
            name=name or f.__name__,
            description=description or doc_description,
            fn=f,
            args_model=_build_args_model(f, param_docs),
        )

    return wrap(fn) if fn is not None else wrap
