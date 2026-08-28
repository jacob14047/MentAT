"""
Prompt template specifico dell'Execution Agent.

Carica il prompt di sistema dal file execution_prompt.txt e costruisce
il messaggio utente a partire dall'ipotesi selezionata dal Planning Agent
e dai parametri chiave derivati dal Recon Report.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from prompts.base_prompt import BasePromptTemplate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PROMPT_FILE = _PROJECT_ROOT / "prompts" / "execution_prompt.txt"

_EXECUTION_OUTPUT_SCHEMA = {
    "selected_hypothesis": {
        "id": "int",
        "rank": "int",
        "title": "string",
        "description": "string",
    },
    "attack_config": {
        "attack_type": "string - tipo dalla mappa (intercept_resend, pns, blinding, trojan_horse, qber_tamper, channel_noise, source_attack)",
        "interception_rate": "float 0.0-1.0",
        "eve_pns_enabled": "boolean",
        "eve_pns_block_ratio": "float 0.0-1.0",
        "blinding_attack_active": "boolean",
        "trojan_horse_prob": "float 0.0-1.0",
        "qber_tamper_active": "boolean",
        "qber_tamper_amount": "float 0.0-1.0",
        "depolarization_prob": "float 0.0-1.0",
    },
    "channel_params": {
        "mean_photon_num": "float - parametro fisso della sorgente laser di Alice",
        "amplitude_damping_gamma": "float - parametro fisso: perdita del canale",
        "distance_km": "float - parametro fisso: distanza del collegamento",
    },
    "optimization_rationale": "string - spiegazione delle scelte di ottimizzazione (5-8 frasi)",
    "expected_qber_increase": "float",
    "expected_eve_info": "float",
    "security_threshold_margin": "float",
    "confidence": "float 0.0-1.0",
}


class ExecutionPromptTemplate(BasePromptTemplate):
    @property
    def system_prompt(self) -> str:
        return _PROMPT_FILE.read_text(encoding="utf-8")

    def render(self, context: Dict[str, Any]) -> list:
        recon_params = context.get("recon_params", {})
        system = _PROMPT_FILE.read_text(encoding="utf-8")
        system = system.format(
            mean_photon_num=recon_params.get("mean_photon_num", "N/A"),
            amplitude_damping_gamma=recon_params.get("amplitude_damping_gamma", "N/A"),
            distance_km=recon_params.get("distance_km", "N/A"),
            detector_efficiency=recon_params.get("detector_efficiency", "N/A"),
            dark_count_rate=recon_params.get("dark_count_rate", "N/A"),
            channel_loss=recon_params.get("channel_loss", "N/A"),
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": self.build_user_prompt(context)},
        ]

    def build_user_prompt(self, context: Dict[str, Any]) -> str:
        hypothesis = context.get("hypothesis", {})
        recon_params = context.get("recon_params", {})
        abort_threshold = context.get("abort_threshold", 0.11)

        return (
            "## Ipotesi Selezionata dal Planning Agent (rank=1)\n"
            f"**title:** {hypothesis.get('title', 'N/A')}\n\n"
            f"**description:** {hypothesis.get('description', 'N/A')}\n\n"
            f"**plausibility_score:** {hypothesis.get('plausibility_score', 'N/A')}\n\n"
            f"**supporting_evidence:** {json.dumps(hypothesis.get('supporting_evidence', []), indent=2, ensure_ascii=False)}\n\n"
            "## Parametri Chiave dal Recon Report\n"
            f"{json.dumps(recon_params, indent=2, ensure_ascii=False)}\n\n"
            "## Contesto di Protocollo\n"
            f"**abort_threshold (QBER):** {abort_threshold}\n\n"
            "## Istruzioni\n"
            "1. Mappa il titolo dell'ipotesi al tipo di attacco usando la mappa\n"
            "   fornita nel prompt di sistema.\n"
            "2. Calcola i parametri ottimali usando le formule di ottimizzazione.\n"
            "3. Assicurati che expected_qber_increase < (abort_threshold - current_qber).\n"
            "4. Rispondi ESCLUSIVAMENTE con un oggetto JSON conforme a questo schema:\n\n"
            f"{json.dumps(_EXECUTION_OUTPUT_SCHEMA, indent=2, ensure_ascii=False)}"
        )
