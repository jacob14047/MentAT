"""
Prompt template specifico del Planning Agent.

Carica il prompt di sistema dal file planning_prompt.txt e costruisce
il messaggio utente a partire dal report del Recon Agent e dai parametri
grezzi del canale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from prompts.base_prompt import BasePromptTemplate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "planning_prompt.txt"

_PLANNING_OUTPUT_SCHEMA = {
    "hypotheses": [
        {
            "id": "int - identificativo univoco",
            "rank": "int - posizione nel ranking (1 = piu' promettente)",
            "title": "string - nome sintetico dell'ipotesi d'attacco",
            "description": "string - descrizione dettagliata (3-6 frasi)",
            "supporting_evidence": ["string - evidenza dal report Recon"],
            "plausibility_score": "float 0.0-1.0",
            "required_recon_data": ["string - dato aggiuntivo necessario"],
            "next_steps": ["string - passo per Execution Agent"],
        }
    ],
    "ranking_rationale": "string - spiegazione del ranking (5-8 frasi)",
    "overall_risk_assessment": "string - valutazione complessiva del rischio (3-5 frasi)",
    "confidence": "float 0.0-1.0",
}


class PlanningPromptTemplate(BasePromptTemplate):
    @property
    def system_prompt(self) -> str:
        return _PROMPT_FILE.read_text(encoding="utf-8")

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        recon_analysis = context.get("recon_analysis", {})
        raw_parameters = context.get("raw_parameters", {})
        metadata = context.get("metadata", {})

        weaknesses = recon_analysis.get("exploitable_weaknesses", [])
        focus_areas = recon_analysis.get("recommended_focus_areas_for_planning", [])
        constraints = recon_analysis.get("constraints_on_eve", [])

        return (
            "## Report del Recon Agent (contesto compresso)\n"
            f"**attack_surface_summary:** {recon_analysis.get('attack_surface_summary', 'N/A')}\n\n"
            f"**freedom_of_action_score:** {recon_analysis.get('freedom_of_action_score', 'N/A')}\n\n"
            f"**confidence:** {recon_analysis.get('confidence', 'N/A')}\n\n"
            f"**rationale:** {recon_analysis.get('rationale', 'N/A')}\n\n"
            "### Debolezze sfruttabili\n"
            f"{json.dumps(weaknesses, indent=2, ensure_ascii=False)}\n\n"
            "### Aree focali per il planning\n"
            f"{json.dumps(focus_areas, indent=2, ensure_ascii=False)}\n\n"
            "### Vincoli su Eve\n"
            f"{json.dumps(constraints, indent=2, ensure_ascii=False)}\n\n"
            "## Parametri chiave del canale\n"
            f"{json.dumps(raw_parameters, indent=2, ensure_ascii=False)}\n\n"
            "## Metadati\n"
            f"{json.dumps(metadata, indent=2, ensure_ascii=False)}\n\n"
            "## Istruzioni\n"
            "Analizza il report del Recon Agent sopra e genera una lista di "
            "ipotesi d'attacco ordinate per plausibilita'. Per ciascuna ipotesi, "
            "cita esplicitamente le evidenze dal report Recon che la supportano.\n\n"
            "Devi rispondere ESCLUSIVAMENTE con un oggetto JSON conforme a questo schema:\n\n"
            f"{json.dumps(_PLANNING_OUTPUT_SCHEMA, indent=2, ensure_ascii=False)}"
        )
