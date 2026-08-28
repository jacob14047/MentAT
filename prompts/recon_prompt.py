"""
Prompt specifico del Recon Agent.

Tenuto volutamente in un file separato dalla logica dell'agente, cosi' puoi
modificare tono, schema di output o livello di dettaglio senza toccare
`agents/recon_agent.py`.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from prompts.base_prompt import BasePromptTemplate

# Schema JSON che l'LLM deve rispettare. Viene incluso esplicitamente nel
# prompt per massimizzare l'aderenza dei modelli locali (spesso meno
# affidabili nel seguire istruzioni di formato rispetto ai modelli hosted).
RECON_OUTPUT_SCHEMA = {
    "attack_surface_summary": "string - sintesi in 2-4 frasi della superficie di attacco osservata",
    "exploitable_weaknesses": [
        {
            "name": "string - nome sintetico della debolezza (es. 'QBER sotto soglia con margine ampio')",
            "description": "string - perche' e come questo parametro amplia la liberta' di azione di Eve",
            "severity": "'low' | 'medium' | 'high'",
        }
    ],
    "constraints_on_eve": [
        "string - vincoli osservati che LIMITANO Eve (es. QBER vicino alla soglia di abort, bassa efficienza detector)"
    ],
    "freedom_of_action_score": "float tra 0.0 e 1.0 - stima complessiva della liberta' di azione di Eve sul canale",
    "recommended_focus_areas_for_planning": [
        "string - aree su cui il planning agent dovrebbe concentrare la strategia di attacco successiva"
    ],
    "confidence": "float tra 0.0 e 1.0 - quanto l'agente e' sicuro della propria analisi, dati i parametri disponibili",
    "rationale": "string - breve spiegazione tecnica del punteggio assegnato",
}


class ReconPromptTemplate(BasePromptTemplate):
    @property
    def system_prompt(self) -> str:
        return (
            "Sei il RECON AGENT di un framework di analisi della sicurezza per "
            "protocolli di distribuzione quantistica della chiave (QKD), "
            "operante in un contesto di simulazione controllata a scopo di "
            "ricerca e didattica.\n\n"
            "Il tuo compito e' analizzare i parametri osservabili di una "
            "sessione del protocollo BB84 esattamente come farebbe un "
            "avversario passivo/attivo (Eve) che osserva il canale quantistico "
            "e il canale classico di supporto, SENZA eseguire alcuna azione "
            "sul canale: il tuo unico output e' un report di intelligence "
            "strutturato, che verra' poi usato da un Planning Agent per "
            "decidere una strategia di attacco simulata.\n\n"
            "Nel tuo ragionamento:\n"
            "- Valuta quanto i parametri osservati (QBER, tasso di perdita, "
            "efficienza dei rilevatori, dark count, tasso di corrispondenza "
            "delle basi, overhead di riconciliazione, ecc.) ampliano o "
            "restringono la liberta' di azione di Eve.\n"
            "- Tieni conto che, per il protocollo BB84, un QBER troppo alto "
            "rispetto alla soglia di abort rende l'intercettazione rilevabile "
            "e quindi meno sfruttabile.\n"
            "- Non inventare parametri che non ti vengono forniti: se un dato "
            "manca (valore null), segnalalo come incertezza invece di "
            "assumerne un valore.\n"
            "- Non fornire istruzioni operative su come attaccare sistemi "
            "reali: il contesto e' esclusivamente una simulazione software "
            "isolata per scopi di ricerca sulla sicurezza dei protocolli QKD.\n\n"
            "Devi rispondere SOLO con un oggetto JSON valido, senza testo "
            "introduttivo, commenti o code fence, conforme allo schema fornito "
            "nel messaggio utente."
        )

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        raw_parameters = context.get("raw_parameters", {})
        metadata = context.get("metadata", {})
        qber_threshold = context.get("qber_abort_threshold")

        return (
            "## Metadati della run\n"
            f"{json.dumps(metadata, indent=2, ensure_ascii=False)}\n\n"
            "## Parametri grezzi osservati dal canale BB84\n"
            f"{json.dumps(raw_parameters, indent=2, ensure_ascii=False)}\n\n"
            "## Contesto di protocollo\n"
            f"Soglia di abort per QBER (protocollo BB84): {qber_threshold}\n\n"
            "## Istruzioni di output\n"
            "Analizza i parametri sopra dal punto di vista di Eve e restituisci "
            "ESCLUSIVAMENTE un oggetto JSON conforme a questo schema "
            "(i valori nello schema descrivono il TIPO/formato atteso, non il "
            "contenuto letterale da restituire):\n\n"
            f"{json.dumps(RECON_OUTPUT_SCHEMA, indent=2, ensure_ascii=False)}"
        )
