# BB84 Recon Framework

Framework modulare per un **Recon Agent** basato su LLM che analizza l'output del
tuo simulatore `bb84_simulator_Eve.py` come farebbe Eve: raccoglie tutti i
parametri osservabili del canale quantistico, li fa interpretare a un modello
linguistico locale (via **LM Studio**) e produce un **report strutturato**,
salvato in un database, che in futuro alimenterà un **Planning Agent**.

L'architettura è ispirata a [VulnBot](https://github.com/KHenryAegis/VulnBot)
(framework multi-agente per penetration testing con LLM), di cui riprende la
divisione concettuale in:

| VulnBot        | Questo progetto | Ruolo |
|----------------|-----------------|-------|
| `roles/`       | `agents/`       | Classi agente (Recon, in futuro Planning, Executor...) |
| `prompts/`     | `prompts/`      | Template di prompt separati dal codice |
| `config/`      | `config/`       | Configurazione LLM / DB / paths |
| `db/`          | `db/`           | Persistenza dei risultati (repository pattern) |
| Kali / target  | `channel/`      | "Bersaglio" da ispezionare: qui è il canale BB84 invece di una macchina |
| `pentest.py`   | `cli.py`        | Entry point che orchestra gli agenti |

## Struttura del progetto

```
bb84_recon_framework/
├── cli.py                      # entry point
├── requirements.txt
├── config/
│   ├── settings.py              # carica config.yaml, espone oggetti di config tipizzati
│   └── config.yaml               # LM Studio host/porta/modello, path DB, soglie BB84
├── prompts/
│   ├── base_prompt.py            # BasePromptTemplate (generica)
│   └── recon_prompt.py           # ReconPromptTemplate (prompt specifico, isolato)
├── llm/
│   ├── base_client.py            # BaseLLMClient (interfaccia generica)
│   └── lmstudio_client.py        # LMStudioClient (implementazione concreta via LM Studio)
├── channel/
│   ├── base_channel_source.py    # BaseChannelSource (interfaccia generica "fonte dati canale")
│   └── bb84_channel_source.py    # BB84ChannelSource: adapter verso bb84_simulator_Eve.py
├── agents/
│   ├── base_agent.py             # BaseAgent (generico: LLM + prompt + repository)
│   └── recon_agent.py            # ReconAgent (specifico: raccolta parametri + analisi Eve)
├── db/
│   ├── models.py                 # ReconReport (modello dati del risultato)
│   ├── base_repository.py        # BaseReconRepository (interfaccia generica)
│   └── sqlite_repository.py      # SQLiteReconRepository (implementazione concreta)
├── utils/
│   ├── logger.py
│   └── json_utils.py              # parsing robusto di JSON prodotto dall'LLM
└── data/
    └── recon_reports.db           # generato a runtime (SQLite)
```

## Perché questa suddivisione

Ogni layer ha **una sola responsabilità** ed è definito prima come classe
astratta generica, poi specializzato:

1. **`channel/`** — "cosa osservo". Non sa nulla di LLM né di DB. Sa solo
   estrarre parametri grezzi dal simulatore (QBER, tasso di scarto, perdite,
   strategia di Eve, ecc.) e normalizzarli in un dizionario.
2. **`llm/`** — "come parlo a un modello". Non sa nulla di BB84. È un client
   HTTP generico verso un endpoint chat-completions compatibile OpenAI (quello
   esposto da LM Studio). Domani potrai aggiungere `OpenAIClient`,
   `OllamaClient`, ecc. senza toccare il resto.
3. **`prompts/`** — "cosa chiedo al modello". Puramente testuale/dichiarativo,
   separato dal codice come richiesto. Il `ReconPromptTemplate` definisce
   sia le istruzioni di sistema (ruolo "Eve/recon") sia lo schema JSON atteso
   in output.
4. **`agents/`** — "chi orchestra". Il `BaseAgent` sa come chiamare un LLM con
   un prompt e validare/parsare il JSON di risposta. Il `ReconAgent` aggiunge
   la logica specifica: prende i parametri dal `channel`, li passa al prompt,
   ottiene l'analisi e la fa persistere.
5. **`db/`** — "dove conservo il risultato". Interfaccia generica
   (`BaseReconRepository`) + implementazione SQLite. Il Planning Agent futuro
   leggerà da qui senza dover sapere come i dati sono stati generati.

Questo significa che, quando implementerai il **Planning Agent**, ti basterà:
- creare `agents/planning_agent.py` che eredita da `BaseAgent`,
- creare `prompts/planning_prompt.py`,
- fargli leggere gli ultimi `ReconReport` da `SQLiteReconRepository`.

## Configurare LM Studio

1. Apri LM Studio, carica un modello e avvia il **Local Server** (tab
   "Developer" → "Start Server"). Di default espone un'API compatibile OpenAI
   su `http://localhost:1234/v1`.
2. Modifica `config/config.yaml` con l'host/porta/nome modello corretti (il
   nome modello lo trovi nella tab server di LM Studio, es.
   `"qwen2.5-7b-instruct"`).

## Collegare il tuo `bb84_simulator_Eve.py`

Il file `channel/bb84_channel_source.py` contiene la classe
`BB84ChannelSource`, che **devi adattare** ai nomi reali di attributi/metodi
del tuo simulatore (non avendo il file non posso conoscerli). Ho lasciato:

- un fallback che funziona già oggi con un semplice **dizionario di
  parametri** o un **JSON di risultati** salvato su disco;
- un punto di estensione chiaro (`_extract_from_simulator_instance`) con i
  `TODO` da riempire con i tuoi nomi reali (es. `sim.qber`,
  `sim.get_stats()`, `sim.eve.strategy`, ...).

## Esecuzione

```bash
pip install -r requirements.txt
python cli.py --source dict --input path/ai/tuoi/risultati.json
# oppure, se importi direttamente il simulatore:
python cli.py --source simulator
```

Il report finale (parametri grezzi + analisi LLM + punteggio di "libertà
d'azione" di Eve) viene stampato a schermo e salvato nella tabella
`recon_reports` del database SQLite in `data/recon_reports.db`.
