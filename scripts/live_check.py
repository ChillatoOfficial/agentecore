"""Prova i provider con chiamate REALI (usano le tue chiavi e consumano token).

Le chiavi si leggono solo dalle variabili d'ambiente (OPENAI_API_KEY, ANTHROPIC_API_KEY,
GOOGLE_API_KEY). Un provider senza chiave o senza modello viene saltato.

    python scripts/live_check.py --openai-model <modello> --anthropic-model <modello> --google-model <modello>

Per ogni provider controlla: testo, streaming, tool call, parametri dei tool, immagine, async.
Esce con codice 1 se qualcosa fallisce.
"""
import argparse
import asyncio
import base64
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentecore import Agent, Anthropic, Google, Media, MediaBlock, OpenAI, TextBlock, tool  # noqa: E402

# PNG 1x1 rosso
PNG_1X1 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@tool
def somma(a: int, b: int) -> int:
    """Somma due numeri interi.

    Args:
        a: Primo addendo.
        b: Secondo addendo.
    """
    return a + b


@tool
def ora() -> str:
    """Restituisce l'ora corrente (tool senza parametri)."""
    return "12:00"


def check_text(client):
    text = client.invoke("Rispondi solo con la parola: ciao").text
    assert text.strip(), "risposta vuota"


def check_stream(client):
    chunks, done = [], None
    for event in client.stream_invoke("Conta da 1 a 5, separati da virgole."):
        if event.type == "text":
            chunks.append(event.text)
        else:
            done = event.response
    assert done is not None and done.text.strip(), "nessuna risposta finale"
    assert "".join(chunks) == done.text, "i frammenti non coincidono con il testo finale"


def check_tool(client):
    agent = Agent(client, tools=[somma, ora])
    text = agent.run("Usa il tool somma per calcolare 17 + 25 e dimmi il risultato.").text
    assert "42" in text, f"risultato inatteso: {text!r}"


def check_tool_stream(client):
    agent = Agent(client, tools=[somma])
    events = list(agent.stream("Usa il tool somma per calcolare 100 + 23."))
    assert any(e.type == "tool" for e in events), "il tool non è stato chiamato"
    assert "123" in events[-1].response.text, "risultato inatteso"


def check_image(client):
    blocks = [
        TextBlock(content="Di che colore è questa immagine? Rispondi in una parola."),
        MediaBlock(media=Media.from_bytes(base64.b64decode(PNG_1X1), "image/png")),
    ]
    assert client.invoke(blocks).text.strip(), "risposta vuota"


async def check_async(client):
    agent = Agent(client, tools=[somma])
    try:
        result = await agent.arun("Usa il tool somma per calcolare 1 + 1.")
    finally:
        await client.aclose()
    assert "2" in result.text, f"risultato inatteso: {result.text!r}"


def run(name, fn, client):
    try:
        result = fn(client)
        if asyncio.iscoroutine(result):
            asyncio.run(result)
        print(f"  PASS  {fn.__name__}")
        return True
    except Exception:
        print(f"  FAIL  {fn.__name__}")
        traceback.print_exc(limit=3)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openai-model")
    parser.add_argument("--anthropic-model")
    parser.add_argument("--google-model")
    args = parser.parse_args()

    providers = [
        ("OpenAI", args.openai_model, "OPENAI_API_KEY", OpenAI),
        ("Anthropic", args.anthropic_model, "ANTHROPIC_API_KEY", Anthropic),
        ("Google", args.google_model, "GOOGLE_API_KEY", Google),
    ]
    failures = 0
    ran = 0
    for name, model, env, cls in providers:
        if not model or not os.environ.get(env):
            print(f"{name}: saltato (serve --{name.lower()}-model e la variabile {env})")
            continue
        ran += 1
        print(f"{name} ({model})")
        client = cls(model)
        for check in (check_text, check_stream, check_tool, check_tool_stream, check_image, check_async):
            failures += not run(name, check, client)
        client.close()
    if not ran:
        print("Nessun provider provato.")
        return 2
    print(f"\n{'TUTTO OK' if not failures else f'{failures} controlli falliti'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
