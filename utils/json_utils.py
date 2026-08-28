"""
I modelli locali serviti da LM Studio non garantiscono un JSON puro:
a volte lo circondano con ```json ... ``` o con frasi introduttive.
Questa utility isola e valida l'oggetto JSON in modo tollerante.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict


class JSONExtractionError(ValueError):
    """Sollevata quando non e' stato possibile ricavare un JSON valido dal testo."""


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise JSONExtractionError("La risposta del modello e' vuota.")

    # 1. Prova diretta: magari e' gia' JSON puro
    candidate = raw_text.strip()
    parsed = _try_parse(candidate)
    if parsed is not None:
        return parsed

    # 2. Rimuovi eventuali code fence ```json ... ``` o ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL)
    if fence_match:
        parsed = _try_parse(fence_match.group(1).strip())
        if parsed is not None:
            return parsed

    # 3. Cerca il primo blocco { ... } bilanciato nel testo
    brace_start = raw_text.find("{")
    if brace_start != -1:
        depth = 0
        for i in range(brace_start, len(raw_text)):
            if raw_text[i] == "{":
                depth += 1
            elif raw_text[i] == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try_parse(raw_text[brace_start : i + 1])
                    if parsed is not None:
                        return parsed
                    break

    raise JSONExtractionError(
        f"Nessun JSON valido trovato nella risposta del modello:\n{raw_text}"
    )


def _try_parse(text: str) -> Dict[str, Any] | None:
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None
