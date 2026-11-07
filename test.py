"""Prova manuale dell'agente con OpenRouter.

Imposta la chiave nell'ambiente prima di eseguirlo:
    PowerShell:  $env:OPENROUTER_API_KEY = "..."
    bash:        export OPENROUTER_API_KEY=...
"""

import os
import sys

from agentecore import Agent, OpenAI, tool

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    sys.exit("Manca la variabile d'ambiente OPENROUTER_API_KEY.")


@tool
def meteo(city: str) -> str:
    """Restituisce il meteo di una città."""
    return f"a {city} fanno 17 gradi"


agent = Agent(
    OpenAI(
        "nvidia/nemotron-3.5-lightning:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    ),
    tools=[meteo],
)

response = agent.run("che tempo fa a roma?")
print("L'agente ha risposto:", response.text)
for c in response.tool_calls:
    print(f"Tool usato: {c.name}({c.arguments}) -> {c.result}")
