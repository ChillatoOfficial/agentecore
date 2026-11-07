from typing import Any

# Parole chiave dello schema che Gemini accetta nei parametri delle funzioni.
_GEMINI_KEYS = frozenset({
    "type", "format", "description", "nullable", "enum", "items", "properties",
    "required", "minItems", "maxItems", "minimum", "maximum", "anyOf",
})
_MAX_DEPTH = 32


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Riduce uno schema JSON (come quello di Pydantic) al sottoinsieme accettato da Gemini.

    Risolve i `$ref`, trasforma `Optional[X]` in `nullable`, converte `const` in `enum`
    e scarta le parole chiave non supportate (`default`, `additionalProperties`, `title`, ...).
    """
    return _convert(schema, schema.get("$defs", {}), 0)


def _convert(node: Any, defs: dict[str, Any], depth: int) -> Any:
    if not isinstance(node, dict):
        return node
    if depth > _MAX_DEPTH:
        raise ValueError("Schema troppo annidato o ricorsivo: Gemini non lo supporta")

    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name not in defs:
            raise ValueError(f"Riferimento allo schema non risolvibile: {node['$ref']}")
        node = {**defs[name], **{k: v for k, v in node.items() if k != "$ref"}}

    nullable = bool(node.get("nullable"))

    if "const" in node:
        node = {**node, "enum": [node["const"]]}

    if isinstance(node.get("type"), list):
        types = [t for t in node["type"] if t != "null"]
        nullable = nullable or len(types) < len(node["type"])
        rest = {k: v for k, v in node.items() if k != "type"}
        if len(types) == 1:
            node = {**rest, "type": types[0]}
        else:
            node = {**rest, "anyOf": [{"type": t} for t in types]}

    if "anyOf" in node:
        variants = node["anyOf"]
        non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
        nullable = nullable or len(non_null) < len(variants)
        rest = {k: v for k, v in node.items() if k != "anyOf"}
        if len(non_null) == 1:
            node = {**rest, **_convert(non_null[0], defs, depth + 1)}
        else:
            node = {**rest, "anyOf": [_convert(v, defs, depth + 1) for v in non_null]}

    out: dict[str, Any] = {}
    for key, value in node.items():
        if key not in _GEMINI_KEYS:
            continue
        if key == "properties":
            out[key] = {k: _convert(v, defs, depth + 1) for k, v in value.items()}
        elif key == "items":
            out[key] = _convert(value, defs, depth + 1)
        else:
            out[key] = value
    if nullable:
        out["nullable"] = True
    return out
