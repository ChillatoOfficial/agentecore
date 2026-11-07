# agentecore

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Licenza: MIT](https://img.shields.io/badge/licenza-MIT-green)](LICENSE)
![Stato: alpha](https://img.shields.io/badge/stato-alpha-orange)

Un piccolo framework Python per costruire **agenti** e **chatbot** basati su LLM, con la stessa interfaccia per OpenAI, Anthropic e Google Gemini.

```python
from agentecore import Agent, OpenAI, tool

@tool
def somma(a: int, b: int) -> int:
    """Somma due numeri."""
    return a + b

agent = Agent(OpenAI("<nome-modello>"), tools=[somma])
print(agent.run("Quanto fa 17 + 25?").text)
```

Per cambiare provider cambi solo il client. Due dipendenze in tutto: `httpx` e `pydantic`.

> [!NOTE]
> Il progetto è in fase **alpha**: l'API può cambiare tra una versione e l'altra.

## Indice

- [Funzionalità](#funzionalità)
- [Installazione](#installazione)
- [Chiavi API](#chiavi-api)
- [Guida rapida](#guida-rapida)
- [Agente](#agente)
- [Tool](#tool)
- [Vedere i tool usati](#vedere-i-tool-usati)
- [Streaming](#streaming)
- [Immagini e PDF](#immagini-e-pdf)
- [Async e prestazioni](#async-e-prestazioni)
- [Memoria persistente](#memoria-persistente)
- [Aggiungere un provider](#aggiungere-un-provider)
- [Limiti noti](#limiti-noti)
- [Contribuire](#contribuire)

## Funzionalità

- **Più provider**: OpenAI e qualunque API compatibile (Ollama, vLLM, OpenRouter… tramite `base_url`), Anthropic e Google Gemini
- **Tool** con il decoratore `@tool`: lo schema viene generato dai type hint e dal docstring
- **Tool usati visibili**: nome, argomenti, risultato, esito e durata di ogni tool eseguito
- **Streaming** del testo, sincrono e asincrono, anche mentre l'agente usa i tool
- **Async nativo**: connessioni riutilizzate, retry con backoff, tool call in parallelo
- **Immagini e PDF** come input
- **Memoria** della conversazione, in RAM o su SQLite
- **Tipizzato** (`py.typed`)

## Installazione

Serve Python 3.10 o successivo. Il pacchetto non è ancora su PyPI: si installa da GitHub.

```bash
pip install git+https://github.com/<utente>/agentecore.git
```

Oppure da una copia locale:

```bash
git clone https://github.com/<utente>/agentecore.git
cd agentecore
pip install .
```

## Chiavi API

Le chiavi si passano al costruttore (`api_key="..."`) oppure si leggono dalle variabili d'ambiente:

| Provider  | Variabile d'ambiente |
|-----------|----------------------|
| OpenAI    | `OPENAI_API_KEY`     |
| Anthropic | `ANTHROPIC_API_KEY`  |
| Google    | `GOOGLE_API_KEY`     |

> [!WARNING]
> Non scrivere le chiavi nel codice e non committarle: usa le variabili d'ambiente o un file `.env` escluso da git.

## Guida rapida

Al posto di `<nome-modello>` metti il nome di un modello del provider che usi.

```python
from agentecore import Agent, Anthropic, Google, OpenAI

# solo il client, senza agente
print(OpenAI("<nome-modello>").invoke("Ciao!").text)

# lo stesso agente con provider diversi
agent = Agent(Anthropic("<nome-modello>"), system_prompt="Sei un assistente utile.")
agent = Agent(Google("<nome-modello>"), system_prompt="Sei un assistente utile.")

# un server compatibile con OpenAI, per esempio Ollama in locale
client = OpenAI("<nome-modello>", base_url="http://localhost:11434/v1", api_key="ollama")
```

## Agente

`Agent` ripete il ciclo «il modello risponde → esegui i tool richiesti → rimanda i risultati» finché il modello non dà una risposta senza tool.

```python
Agent(
    client,                       # OpenAI, Anthropic, Google o un tuo Client
    tools=[somma],                # funzioni decorate con @tool
    system_prompt="...",          # istruzioni di sistema
    max_steps=10,                 # numero massimo di giri del ciclo
    memory=None,                  # una Memory (anche SQLiteMemory); per default in RAM
    on_tool=None,                 # funzione chiamata dopo ogni tool eseguito
)
```

| Metodo | Restituisce |
|--------|-------------|
| `run(input)` / `arun(input)` | la risposta intera (`ClientResponse` con `text`, `usage`, `tool_calls`) |
| `stream(input)` / `astream(input)` | eventi `text`, `tool` e `done` man mano che arrivano |

`input` è una stringa o una lista di blocchi (testo, immagini, PDF). Se il modello non finisce entro `max_steps` viene sollevato un `RuntimeError`.

## Tool

Lo schema del tool viene generato dalla funzione. Le descrizioni dei parametri si scrivono nella sezione `Args:` del docstring oppure con `Annotated`:

```python
from typing import Annotated, Literal
from pydantic import Field
from agentecore import tool

@tool
def meteo(
    citta: str,
    unita: Literal["C", "F"] = "C",
    giorni: Annotated[int, Field(description="Giorni di previsione", ge=1, le=7)] = 1,
) -> str:
    """Restituisce il meteo di una città.

    Args:
        citta: Nome della città.
        unita: Unità di misura della temperatura.
    """
    ...
```

- I tool possono essere `def` o `async def`. I tool `async def` richiedono `arun()` o `astream()`.
- Gli argomenti inviati dal modello vengono validati. Se qualcosa va storto (argomenti sbagliati, eccezione nel tool, tool inesistente) l'errore torna al modello come risultato, così può correggersi.
- Per Gemini lo schema viene adattato automaticamente al sottoinsieme che accetta. Gli schemi ricorsivi non sono supportati.

## Vedere i tool usati

Ogni tool eseguito è un `ToolExecution` con `name`, `arguments`, `result`, `error` e `duration` (in secondi). Puoi leggerli in tre modi:

```python
# 1. a fine esecuzione, in ordine
risposta = agent.run("Quanto fa 17 + 25?")
for c in risposta.tool_calls:
    print(c.name, c.arguments, "->", c.result)

# 2. in diretta, anche con run() e arun()
agent = Agent(client, tools=[somma], on_tool=lambda c: print("tool:", c.name, c.arguments))

# 3. nello streaming, con l'evento "tool" (vedi sotto)
```

- `error` è vero se il tool non esiste o ha sollevato un'eccezione; il messaggio sta in `result` ed è lo stesso testo che riceve il modello.
- `tool_calls` contiene i tool dell'intera richiesta, non solo dell'ultimo passo. Se la richiesta fallisce (per esempio supera `max_steps`) la risposta non esiste: usa `on_tool` o lo streaming per non perdere la cronologia.
- I tool chiesti nello stesso passo girano in parallelo con `arun()`/`astream()` e in sequenza con `run()`/`stream()`.

## Streaming

```python
for event in agent.stream("Raccontami una storia breve"):
    if event.type == "text":
        print(event.text, end="", flush=True)     # frammenti di testo
    elif event.type == "tool":
        c = event.call                             # un tool è stato eseguito
        print(f"\n[{c.name}({c.arguments}) -> {c.result} in {c.duration:.2f}s]")
    elif event.type == "done":
        print("\n", event.response.usage)          # risposta completa
```

La versione asincrona è `async for event in agent.astream(...)`. Anche i client hanno `stream_invoke` e `a_stream_invoke`.

## Immagini e PDF

```python
from agentecore import Media, MediaBlock, TextBlock

risposta = client.invoke([
    TextBlock(content="Che cosa c'è in questa immagine?"),
    MediaBlock(media=Media.from_path("foto.png")),
])
```

Gli allegati si creano con `Media.from_path`, `Media.from_bytes(dati, mime_type)` e `Media.from_url`. Formati supportati: PNG, JPEG, GIF, WebP e PDF.

| Provider  | Immagine da file | Immagine da URL | PDF da file | PDF da URL |
|-----------|:----------------:|:---------------:|:-----------:|:----------:|
| OpenAI    | ✅ | ✅ | ✅ | ❌ |
| Anthropic | ✅ | ✅ | ✅ | ✅ |
| Gemini    | ✅ | ❌ | ✅ | ❌ |

Dove l'URL non è supportato viene sollevato un `ValueError`: scarica il file e usa `from_bytes`.

## Async e prestazioni

```python
import asyncio
from agentecore import Agent, OpenAI

domande = ["Quanto fa 2 + 2?", "Qual è la capitale della Francia?"]

async def main():
    async with OpenAI("<nome-modello>") as client:
        risposte = await asyncio.gather(*(Agent(client).arun(d) for d in domande))
        for r in risposte:
            print(r.text)

asyncio.run(main())
```

- **Riusa il client**: ogni client tiene un pool di connessioni keep-alive. Un client si può condividere tra più agenti; un agente invece ha la sua memoria, quindi crea un agente per conversazione.
- **Retry** con backoff esponenziale su errori di rete e sugli stati 408, 409, 429, 5xx e 529, rispettando `Retry-After`. Nello streaming il retry avviene solo prima del primo byte, per non duplicare il testo.
- **Errori leggibili**: un errore HTTP include il corpo della risposta del provider.
- **Opzioni**: `OpenAI("<nome-modello>", timeout=30, connect_timeout=5, max_retries=5, max_connections=200)`.
- Chiudi le risorse con `with` / `async with`, oppure con `close()` / `aclose()`.

## Memoria persistente

Per default la conversazione resta in RAM. Con `SQLiteMemory` viene salvata in un file SQLite (modulo standard di Python, nessuna dipendenza in più):

```python
from agentecore import Agent, OpenAI, SQLiteMemory

with SQLiteMemory("memoria.db", session_id="anna") as memory:
    Agent(OpenAI("<nome-modello>"), memory=memory).run("Mi chiamo Anna.")

# al prossimo avvio la conversazione riprende da dove era rimasta
with SQLiteMemory("memoria.db", session_id="anna") as memory:
    print(Agent(OpenAI("<nome-modello>"), memory=memory).run("Come mi chiamo?").text)
```

- Un file può contenere più sessioni, per esempio una per utente: `list_sessions()`, `delete_session(id)`, `clear()`.
- Ogni messaggio viene scritto subito su disco.
- I dati, allegati compresi, sono salvati **in chiaro** nel file `.db`.
- Non aprire la stessa sessione con due oggetti contemporaneamente.
- La memoria non viene troncata: conversazioni molto lunghe finiranno per superare il contesto del modello.

## Aggiungere un provider

Per un'API HTTP/JSON estendi `HTTPClient`, che gestisce già pool di connessioni, async, retry e chiusura:

```python
from agentecore import ClientResponse, HTTPClient

class MioProvider(HTTPClient):
    def _build_request(self, *, input, tools, memory, temperature, system_prompt, stream=False):
        # converte memoria e input nel formato del provider
        return url, headers, payload

    def _parse_response(self, data) -> ClientResponse:
        # converte il JSON della risposta in blocchi
        ...

    def _new_stream(self):
        # opzionale: un accumulatore con feed() e finish();
        # senza, lo streaming restituisce la risposta intera
        ...
```

Per un trasporto diverso da HTTP estendi direttamente `Client` e implementa `_invoke`.

## Limiti noti

- Allegati: solo immagini e PDF (niente audio o video); i risultati dei tool sono solo testo.
- Nello streaming gli argomenti delle tool call arrivano completi, non a pezzi.
- Nessun troncamento automatico della memoria.
- Uno `StructuredBlock` salvato in SQLite torna come dizionario, non come modello Pydantic.
- Niente HTTP/2.

## Contribuire

Segnalazioni e pull request sono benvenute. Per preparare l'ambiente:

```bash
git clone https://github.com/<utente>/agentecore.git
cd agentecore
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -e ".[dev]"
```

I test girano offline e non richiedono chiavi:

```bash
python -m unittest discover -s tests -t . -v
```

I test simulano le risposte dei provider. Per verificare che le API reali accettino ciò che il framework invia, usa le tue chiavi (consuma pochi token):

```bash
python scripts/live_check.py --openai-model <modello> --anthropic-model <modello> --google-model <modello>
```

Lo script prova testo, streaming, tool call, immagini e async per ogni provider che ha chiave e modello, ed esce con codice 1 se qualcosa fallisce.

## Ringraziamenti

L'architettura è ispirata a [Datapizza AI](https://github.com/datapizza-labs/datapizza-ai).

## Licenza

Distribuito con licenza [MIT](LICENSE).
