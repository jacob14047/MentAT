# BB84 Recon Framework

![Python](https://img.shields.io/badge/Python-3.x-blue)
![OS](https://img.shields.io/badge/OS-Linux-green)
![LLM](https://img.shields.io/badge/AI-LM%20Studio-orange)
![DB](https://img.shields.io/badge/DB-SQLite-yellow)

Framework modulare per l'analisi recon del canale BB84 tramite agenti basati su LLM locale (LM Studio). Nel repository sono già implementati gli agenti principali all'interno di `agents/`: `ReconAgent`, `PlanningAgent` e `ExecutionAgent`. Il progetto prende in ingresso i risultati del simulatore (o un dizionario/JSON di parametri osservabili), li invia a un modello linguistico locale per ottenere un'analisi strutturata (report) e salva il risultato in un database SQLite.

Principali obiettivi:
- separazione delle responsabilità (channel / prompts / llm / agents / db)
- output JSON strutturato e validato dall'LLM
- persistenza dei report per pipeline multi-agente già implementata

## Caratteristiche

- Analisi locale con LLM via `LM Studio` (no cloud obbligatorio)
- Adapter per sorgenti di dati del canale BB84 (simulatore o file JSON)
- Prompt template separati in `prompts/`
- Repository pattern per persistere i `ReconReport` in `data/recon_reports.db`
-- Agenti implementati: `ReconAgent`, `PlanningAgent`, `ExecutionAgent` (vedi `agents/`)

## Tech stack

| Componente | Tecnologia |
|-----------:|:----------|
| Linguaggio | Python 3 |
| LLM runner | LM Studio (API compatibile OpenAI) |
| Database | SQLite (file `data/recon_reports.db`) |
| OS target | Linux / qualsiasi OS con Python |

## Struttura del progetto (sintesi)

```
.
├── cli.py                  # entry point
├── config/                 # configurazione (config.yaml, settings.py)
├── prompts/                # template di prompt per gli agenti
├── llm/                    # client per LM Studio (base + implementazione)
├── channel/                # adapter sorgenti canale BB84 (simulator / json)
├── agents/                 # agenti (BaseAgent, ReconAgent, ...)
├── db/                     # modelli e repository (SQLite)
├── utils/                  # logger, json_utils, helper
└── data/                   # database e output generati
```

File rilevanti:
- `cli.py` — entry point che orchestra l'esecuzione
- `config/config.yaml` — host/porta/modello LM Studio, path DB, soglie
- `channel/bb84_channel_source.py` — adattatore verso il tuo simulatore
- `prompts/recon_prompt.py` — prompt template usato dal `ReconAgent`
- `db/sqlite_repository.py` — implementazione concreta della persistenza

## Requisiti

Assicurati di avere Python 3.8+ e `pip` installati. Installa le dipendenze:

```bash
python -m pip install -r requirements.txt
```

## Configurare LM Studio

1. Avvia LM Studio e carica il modello desiderato.
2. Abilita il `Local Server` (tab Developer → Start Server). Di default l'API è compatibile OpenAI su `http://localhost:1234/v1`.
3. Aggiorna `config/config.yaml` con host, porta e nome del modello.

Esempio minimale `config/config.yaml`:

```yaml
llm:
  base_url: "http://localhost:1234/v1"
  model: "nome-modello"
database:
  path: "data/recon_reports.db"
```

## Collegare il tuo simulatore BB84

Il `BB84ChannelSource` in `channel/bb84_channel_source.py` espone due modalità:

- estrazione da un'istanza di simulatore (se importi il tuo codice)
- caricamento da file JSON/dizionario (utile per test)

Adatta `_extract_from_simulator_instance` ai nomi delle proprietà/metodi del tuo simulatore (es. `sim.qber`, `sim.get_stats()`), oppure passa un JSON con i campi attesi.

## Esecuzione

Esempi:

```bash
# usare un file JSON come sorgente
python cli.py --source dict --input path/ai/tuoi/risultati.json

# usare il simulatore importato (se supportato)
python cli.py --source simulator
```

L'output è stampato su console e salvato nella tabella `recon_reports` del file SQLite in `data/recon_reports.db`.

## Sviluppo e estensioni

-- Personalizzare o estendere gli agenti già presenti in `agents/` (recon, planning, execution)
-- Implementare client aggiuntivi in `llm/` per OpenAI/Ollama, se necessario
-- Migliorare la validazione JSON nei template in `prompts/`

## Disclaimer

Questo progetto è inteso per scopi di ricerca ed educazione. Usa sempre strumenti e analisi solo su sistemi di cui possiedi i diritti o per i quali hai autorizzazione scritta. L'autore non è responsabile per uso improprio.

## Autore

Progetto sviluppato come proof-of-concept per analisi recon BB84 usando LLM locali.

## Licenza

Licenza MIT — vedere il file `LICENSE` se presente.

